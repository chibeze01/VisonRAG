from __future__ import annotations

import base64
import io
import logging
from typing import Protocol
from uuid import UUID

from PIL import Image

from visionrag.config import Settings
from visionrag.types import PageResult

logger = logging.getLogger(__name__)


class AnswerGenerator(Protocol):
    def answer(self, question: str, images: list[Image.Image], citations: list[PageResult]) -> str:
        raise NotImplementedError


class GeminiAnswerGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini answer generation.") from exc

        self._client = genai.Client(api_key=api_key)
        self._model = model

    @staticmethod
    def _format_citations(citations: list[PageResult]) -> str:
        lines: list[str] = []
        for row in citations:
            document_id: UUID = row.document_id
            lines.append(
                f"- document_id={document_id}, s3={row.s3_bucket}/{row.s3_key}, "
                f"page={row.page_number}, score={row.score:.4f}"
            )
        return "\n".join(lines)

    def answer(self, question: str, images: list[Image.Image], citations: list[PageResult]) -> str:
        prompt = (
            "You are a document analyst. Answer only from the provided page images. "
            "If the evidence is insufficient, say so. Cite page references in your answer.\n"
            f"Evidence pages:\n{self._format_citations(citations)}\n"
            f"Question: {question}"
        )
        contents = [prompt] + images
        response = self._client.models.generate_content(model=self._model, contents=contents)
        return response.text or ""


class AzureAnswerGenerator:
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for Azure answer generation.") from exc

        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment

    @staticmethod
    def _format_citations(citations: list[PageResult]) -> str:
        lines: list[str] = []
        for row in citations:
            document_id: UUID = row.document_id
            lines.append(
                f"- document_id={document_id}, s3={row.s3_bucket}/{row.s3_key}, "
                f"page={row.page_number}, score={row.score:.4f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{payload}"

    def answer(self, question: str, images: list[Image.Image], citations: list[PageResult]) -> str:
        prompt = (
            "You are a document analyst. Answer only from the provided page images. "
            "If the evidence is insufficient, say so. Cite page references in your answer.\n"
            f"Evidence pages:\n{self._format_citations(citations)}\n"
            f"Question: {question}"
        )

        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image)},
                }
            )

        completion = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": content}],
        )
        if not completion.choices:
            return ""

        message = completion.choices[0].message
        return message.content or ""


class FallbackAnswerGenerator:
    def __init__(self, primary: AnswerGenerator, fallback: AnswerGenerator) -> None:
        self._primary = primary
        self._fallback = fallback

    def answer(self, question: str, images: list[Image.Image], citations: list[PageResult]) -> str:
        try:
            return self._primary.answer(question, images, citations)
        except Exception:
            logger.exception("Primary answer generator failed. Falling back.")
            return self._fallback.answer(question, images, citations)


def create_answer_generator(settings: Settings) -> AnswerGenerator | None:
    azure_configured = bool(
        settings.azure_openai_endpoint and settings.azure_openai_api_key and settings.azure_openai_deployment
    )
    gemini_configured = bool(settings.gemini_api_key)

    azure: AnswerGenerator | None = None
    gemini: AnswerGenerator | None = None

    if azure_configured:
        try:
            azure = AzureAnswerGenerator(
                endpoint=settings.azure_openai_endpoint or "",
                api_key=settings.azure_openai_api_key or "",
                api_version=settings.azure_openai_api_version,
                deployment=settings.azure_openai_deployment or "",
            )
        except Exception:
            logger.exception("Azure answer generator initialization failed.")
    if gemini_configured:
        gemini = GeminiAnswerGenerator(api_key=settings.gemini_api_key or "", model=settings.gemini_model)

    if azure and gemini:
        return FallbackAnswerGenerator(primary=azure, fallback=gemini)
    if azure:
        return azure
    if gemini:
        return gemini
    return None

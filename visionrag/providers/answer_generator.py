from __future__ import annotations

from typing import Protocol
from uuid import UUID

from PIL import Image

from visionrag.types import PageResult


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


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from visionrag.types import PatchBBox, PatchEmbedding


class EmbeddingProvider(Protocol):
    def embed_page(self, image: Image.Image, page_number: int) -> list[PatchEmbedding]:
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError

    def warm(self) -> None:
        raise NotImplementedError


@dataclass
class ColPaliEmbeddingProvider:
    model_name: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None

    def warm(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        try:
            import torch
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError as exc:
            raise RuntimeError(
                "colpali-engine and torch are required. Install project dependencies first."
            ) from exc

        torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model = ColQwen2.from_pretrained(self.model_name, torch_dtype=torch_dtype)
        model = model.to(self.device).eval()
        processor = ColQwen2Processor.from_pretrained(self.model_name)

        self._torch = torch
        self._model = model
        self._processor = processor

    def _to_device(self, batch: dict) -> dict:
        torch = self._torch
        output = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                output[key] = value.to(self.device)
            else:
                output[key] = value
        return output

    def _build_bboxes(self, image: Image.Image, patch_count: int) -> list[PatchBBox]:
        assert self._processor is not None
        assert self._model is not None
        try:
            n_x, n_y = self._processor.get_n_patches(image.size, self._model.spatial_merge_size)
        except TypeError:
            n_x, n_y = self._processor.get_n_patches(image.size, self._model.patch_size)

        if n_x <= 0 or n_y <= 0:
            side = max(1, int(math.sqrt(max(1, patch_count))))
            n_x = side
            n_y = max(1, math.ceil(patch_count / side))

        bboxes: list[PatchBBox] = []
        for i in range(patch_count):
            row = i // n_x
            col = i % n_x
            x1 = col / n_x
            y1 = row / n_y
            x2 = min(1.0, (col + 1) / n_x)
            y2 = min(1.0, (row + 1) / n_y)
            bboxes.append(PatchBBox(x1=x1, y1=y1, x2=x2, y2=y2))
        return bboxes

    def embed_page(self, image: Image.Image, page_number: int) -> list[PatchEmbedding]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        batch = self._processor.process_images([image])
        batch_original = batch
        batch = self._to_device(batch)
        with self._torch.no_grad():
            output = self._model(**batch)
        image_mask = self._processor.get_image_mask(batch_original)
        patch_tensor = output[0][image_mask[0]].detach().cpu()

        bboxes = self._build_bboxes(image=image, patch_count=len(patch_tensor))
        rows: list[PatchEmbedding] = []
        for patch_index, vector in enumerate(patch_tensor):
            rows.append(
                PatchEmbedding(
                    page_number=page_number,
                    patch_index=patch_index,
                    patch_bbox=bboxes[patch_index],
                    embedding=vector.to(dtype=self._torch.float32).tolist(),
                )
            )
        return rows

    def embed_query(self, query: str) -> list[float]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        batch = self._processor.process_queries([query])
        batch = self._to_device(batch)
        with self._torch.no_grad():
            output = self._model(**batch)

        mask = batch["attention_mask"][0].bool()
        token_vectors = output[0][mask].detach().cpu()
        if len(token_vectors) == 0:
            raise RuntimeError("Query embedding returned no tokens.")

        pooled = token_vectors.mean(dim=0)
        return pooled.to(dtype=self._torch.float32).tolist()

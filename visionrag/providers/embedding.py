from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    import torch

import numpy as np
from PIL import Image

from visionrag.Light_merge import LightMerger
from visionrag.types import PatchBBox, PatchEmbedding


def _dense_bbox(bboxes: list[PatchBBox], density_percentile: float) -> PatchBBox:
    """
    Returns a bounding box around the spatially dense core of a patch cluster.

    Rather than the naive union of all member bboxes (which expands to cover outlier
    patches scattered far from the cluster centre), this:
      1. Computes the (cx, cy) centre of each member patch.
      2. Finds the spatial median of those centres — a robust centroid.
      3. Discards patches whose centre is beyond the `density_percentile` quantile
         of the distance distribution from the median (the outliers).
      4. Returns the union bbox of the remaining core patches.

    Args:
        bboxes:              Member patch bboxes for one merged cluster.
        density_percentile:  Fraction in (0, 1] of patches to retain.
                             0.75 keeps the 75% closest to the cluster core.
    """
    if len(bboxes) <= 6:
        # Too few patches to meaningfully filter — return plain union.
        return PatchBBox(
            x1=min(b.x1 for b in bboxes),
            y1=min(b.y1 for b in bboxes),
            x2=max(b.x2 for b in bboxes),
            y2=max(b.y2 for b in bboxes),
        )

    centers = np.array(
        [((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0) for b in bboxes],
        dtype=np.float32,
    )  # (N, 2)

    # Spatial median: robust against outliers unlike the mean
    median_center = np.median(centers, axis=0)  # (2,)
    distances = np.linalg.norm(centers - median_center, axis=1)  # (N,)

    threshold = np.percentile(distances, density_percentile * 100.0)
    core = [bboxes[i] for i, d in enumerate(distances) if d <= threshold]

    # Fallback: if everything is equidistant and threshold rounds to 0, keep all
    if not core:
        core = bboxes

    return PatchBBox(
        x1=min(b.x1 for b in core),
        y1=min(b.y1 for b in core),
        x2=max(b.x2 for b in core),
        y2=max(b.y2 for b in core),
    )


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
    # Optional Light-ColPali token merger applied post-projector (https://arxiv.org/pdf/2506.04997#page=3.56).
    # Set to a LightMerger instance to enable token reduction; None to disable.
    merger: Optional[LightMerger] = field(default=None, repr=True)

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

        if self.merger is not None:
            return self._embed_merged(image, page_number, patch_tensor)
        return self._embed_raw(image, page_number, patch_tensor)

    def _embed_raw(
        self, image: Image.Image, page_number: int, patch_tensor: "torch.Tensor"
    ) -> list[PatchEmbedding]:
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

    def _embed_merged(
        self, image: Image.Image, page_number: int, patch_tensor: "torch.Tensor"
    ) -> list[PatchEmbedding]:
        """
        Apply Light-ColPali token merging (post-projector) as per §3 of the paper.

        Steps:
          1. Build per-patch spatial bboxes for all Np original patches.
          2. Cluster Np patches → Np' clusters via HAC on cosine distance.
          3. Each cluster embedding = mean of its members (paper: §3).
          4. Each cluster bbox = union of its members' bboxes (spatial coverage).
        """
        assert self.merger is not None
        assert self._torch is not None

        # 1. Spatial bboxes for all original patches
        original_bboxes = self._build_bboxes(image=image, patch_count=len(patch_tensor))

        # 2 & 3. Cluster + mean-pool
        merged_tensor, labels, num_clusters = self.merger.merge_with_labels(patch_tensor)

        # 4. Union bbox per cluster + build output
        rows: list[PatchEmbedding] = []
        for cluster_idx in range(num_clusters):
            cluster_id = cluster_idx + 1  # labels are 1-indexed
            member_indices = np.where(labels == cluster_id)[0]
            member_bboxes = [original_bboxes[i] for i in member_indices]

            union_bbox = _dense_bbox(member_bboxes, self.merger.bbox_density_percentile)
            vector = merged_tensor[cluster_idx]
            rows.append(
                PatchEmbedding(
                    page_number=page_number,
                    patch_index=cluster_idx,
                    patch_bbox=union_bbox,
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

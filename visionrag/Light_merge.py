"""
Light-ColPali token merging via hierarchical semantic clustering.

Based on: "Light-ColPali: Efficient Document Retrieval with Token Merging"
https://arxiv.org/pdf/2506.04997

Key findings from the paper:
  - Cosine similarity used to measure patch embedding similarity
  - Hierarchical agglomerative clustering groups Np patches → Np' clusters
  - Each cluster represented by the mean of its member embeddings
  - Merging post-projector yields best performance
  - Merge factor 9x  → 98.2% performance at 11.8% memory
  - Merge factor 25x → 96.3% performance at  3.0% memory
  - Merge factor 49x → 94.6% performance at  1.8% memory
"""

import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity


class LightMerger:
    """
    Merges ColPali/ColQwen2 patch embeddings via hierarchical semantic clustering.

    Given Np post-projector patch embeddings, groups them into Np' clusters
    (where Np' = Np // merge_factor) and replaces each cluster with the mean
    of its member embeddings.

    Args:
        merge_factor: Target reduction ratio Np / Np'. Default 9 gives ~98% NDCG
                      retention at ~12% of original memory (per the paper).
        min_clusters:  Hard floor on Np' to avoid over-merging very short sequences.
        linkage_method: HAC linkage criterion. 'average' is the paper default;
                        'ward' minimises intra-cluster variance and is a reasonable
                        alternative when embeddings are L2-normalised.
    """

    PAPER_BENCHMARKS = {
        9:  {"ndcg_retention": 0.982, "memory_fraction": 0.118},
        25: {"ndcg_retention": 0.963, "memory_fraction": 0.030},
        49: {"ndcg_retention": 0.946, "memory_fraction": 0.018},
    }

    def __init__(
        self,
        merge_factor: int = 9,
        min_clusters: int = 32,
        linkage_method: str = "average",
        bbox_density_percentile: float = 0.75,
    ) -> None:
        if merge_factor < 1:
            raise ValueError(f"merge_factor must be >= 1, got {merge_factor}")
        if not 0.0 < bbox_density_percentile <= 1.0:
            raise ValueError(f"bbox_density_percentile must be in (0, 1], got {bbox_density_percentile}")
        self.merge_factor = merge_factor
        self.min_clusters = min_clusters
        self.linkage_method = linkage_method
        # Fraction of cluster patches (closest to the spatial median) used to
        # compute the output bounding box. Outlier patches beyond this percentile
        # of the distance distribution are excluded from the bbox calculation.
        self.bbox_density_percentile = bbox_density_percentile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Merge patch embeddings into a smaller set of cluster representatives.

        Args:
            patch_embeddings: Float tensor of shape (Np, dim), the post-projector
                              patch embeddings for a single document page.

        Returns:
            Float tensor of shape (Np', dim) where Np' = max(min_clusters, Np // merge_factor).
        """
        merged, _, _ = self.merge_with_labels(patch_embeddings)
        return merged

    def merge_with_labels(
        self, patch_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, np.ndarray, int]:
        """
        Like merge(), but also returns the 1-indexed cluster label array and
        the actual number of clusters so callers can map original patch metadata
        (e.g. spatial bboxes) onto the merged clusters.

        Args:
            patch_embeddings: Float tensor of shape (Np, dim).

        Returns:
            merged:       (Np', dim) mean-pooled cluster embeddings.
            labels:       (Np,) int array; labels[i] is the 1-indexed cluster id
                          for original patch i.
            num_clusters: Np' (actual cluster count after empty-cluster pruning).
        """
        num_patches = patch_embeddings.size(0)
        num_clusters = max(self.min_clusters, num_patches // self.merge_factor)

        if num_patches <= num_clusters:
            # Already at or below target — each patch is its own cluster.
            labels = np.arange(1, num_patches + 1, dtype=np.int32)
            return patch_embeddings, labels, num_patches

        labels = self._cluster(patch_embeddings, num_clusters)
        merged = self._aggregate(patch_embeddings, labels, num_clusters)
        return merged, labels, merged.size(0)

    def __call__(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        return self.merge(patch_embeddings)

    def __repr__(self) -> str:
        return (
            f"LightMerger(merge_factor={self.merge_factor}, "
            f"min_clusters={self.min_clusters}, "
            f"linkage_method='{self.linkage_method}')"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster(self, embeddings: torch.Tensor, num_clusters: int) -> np.ndarray:
        """
        Run hierarchical agglomerative clustering on cosine distance.

        Returns a 1-indexed label array of shape (Np,).
        """
        emb_np = embeddings.detach().cpu().numpy()

        # Cosine similarity in [-1, 1] → distance in [0, 1]
        sim = cosine_similarity(emb_np)
        dist = 1.0 - (sim + 1.0) / 2.0

        # scipy linkage expects a condensed distance matrix
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)

        Z = linkage(condensed, method=self.linkage_method)
        return fcluster(Z, t=num_clusters, criterion="maxclust")

    @staticmethod
    def _aggregate(
        embeddings: torch.Tensor, labels: np.ndarray, num_clusters: int
    ) -> torch.Tensor:
        """
        Replace each cluster with the mean of its member embeddings.

        Per the paper: "Each cluster is then represented by the average of the
        embeddings within it."
        """
        merged = []
        for cluster_id in range(1, num_clusters + 1):
            mask = torch.from_numpy(labels == cluster_id)
            members = embeddings[mask]
            if members.size(0) > 0:
                merged.append(members.mean(dim=0))

        return torch.stack(merged)  # (Np', dim)

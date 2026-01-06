"""Dataset sampling utilities for meta Bayesian optimization tasks."""

from typing import Tuple

import numpy as np
import torch


class TextDataSampler:
    """Samples tasks from preprocessed text datasets.

    Each task corresponds to selecting one target document and a pool of candidate
    documents. Rewards are precomputed as negative L2 distances in the semantic
    embedding space to the target document.
    """

    def __init__(
        self, dataset_path: str, generator: torch.Generator | None = None
    ) -> None:
        # We store small tensors; load safely on CPU.
        data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        self.features: torch.Tensor = data["features"].float()
        self.embeddings: torch.Tensor = data["embeddings"].float()
        self.labels = data.get("labels")
        self.class_names = data.get("class_names")
        self.num_docs = self.features.size(0)
        self.generator = generator

    @property
    def feature_dim(self) -> int:
        return self.features.size(1)

    def sample_task(
        self, num_candidates: int = 500, generator: torch.Generator | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one target and a candidate pool, returning features and rewards."""
        gen = generator if generator is not None else self.generator
        num_candidates = min(num_candidates, self.num_docs - 1)
        indices = torch.randperm(self.num_docs, generator=gen)[: num_candidates + 1]
        target_idx = indices[0]
        candidate_idxs = indices[1:]

        target_emb = self.embeddings[target_idx]
        candidate_embs = self.embeddings[candidate_idxs]
        X_candidates = self.features[candidate_idxs]

        dists = torch.norm(candidate_embs - target_emb, dim=1)
        y_candidates = -dists
        return X_candidates, y_candidates

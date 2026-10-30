"""
Machine learning package containing PyTorch neural modules for memory representation learning.
"""
from ml.contrastive_model import (
    MemoryContrastiveNetwork,
    ContrastiveLoss,
    train_contrastive_step,
    project_embedding,
    evaluate_pair_similarity,
)

__all__ = [
    "MemoryContrastiveNetwork",
    "ContrastiveLoss",
    "train_contrastive_step",
    "project_embedding",
    "evaluate_pair_similarity",
]

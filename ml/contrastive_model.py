"""
PyTorch Contrastive Learning Module for Memory Redundancy Detection and Compression Clustering.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryContrastiveNetwork(nn.Module):
    """
    Feed-forward projection head for learning dense representations specialized
    in detecting semantic memory redundancy and cluster boundaries.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embedding vector into contrastive latent space and L2-normalize.
        Shape: [batch_size, input_dim] -> [batch_size, projection_dim]
        """
        # TODO: Add residual skip connection if projecting larger 768-d or 1024-d embeddings
        proj = self.encoder_mlp(x)
        return F.normalize(proj, p=2, dim=-1)

    def compute_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between inputs in contrastive projection space.
        Shape: [batch_size, input_dim] x [batch_size, input_dim] -> [batch_size]
        """
        p1 = self.forward(x1)
        p2 = self.forward(x2)
        return torch.sum(p1 * p2, dim=-1)


class ContrastiveLoss(nn.Module):
    """
    Triplet / InfoNCE Contrastive Loss:
    Minimizes distance between redundant memory pairs (Anchor, Positive)
    while maximizing distance to unrelated memories (Anchor, Negative).
    """

    def __init__(self, margin: float = 0.5, temperature: float = 0.07) -> None:
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet margin loss.
        """
        # TODO: Implement InfoNCE loss over batch negatives for self-supervised pretraining
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


def train_contrastive_step(
    model: MemoryContrastiveNetwork,
    optimizer: torch.optim.Optimizer,
    anchor_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    neg_embs: torch.Tensor,
    criterion: ContrastiveLoss,
) -> float:
    """
    Perform a single optimization step for fine-tuning memory projections.
    """
    # TODO: Connect this with dataset loader of conversation turn triples
    model.train()
    optimizer.zero_grad()

    proj_a = model(anchor_embs)
    proj_p = model(pos_embs)
    proj_n = model(neg_embs)

    loss = criterion(proj_a, proj_p, proj_n)
    loss.backward()
    optimizer.step()

    return float(loss.item())


def project_embedding(
    model: MemoryContrastiveNetwork,
    embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Project raw vector embedding into contrastive space.
    Accepts 1D/2D list, numpy array, or torch.Tensor.
    Returns normalized torch.Tensor of shape [batch_size, projection_dim].
    """
    model.eval()
    with torch.no_grad():
        if isinstance(embedding, list):
            tensor = torch.tensor(embedding, dtype=torch.float32)
        elif not isinstance(embedding, torch.Tensor):
            tensor = torch.as_tensor(embedding, dtype=torch.float32)
        else:
            tensor = embedding.float()

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        return model(tensor)


def evaluate_pair_similarity(
    model: MemoryContrastiveNetwork,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> float:
    """
    Evaluate cosine similarity between two memory representations in the contrastive space.
    Returns scalar float in [-1.0, 1.0].
    """
    p1 = project_embedding(model, emb1)
    p2 = project_embedding(model, emb2)
    sim = torch.sum(p1 * p2, dim=-1)
    return round(float(sim.squeeze().item()), 4)

"""
Tests for Contrastive Learning Module (ml/contrastive_model.py):
1. Similar memory pairs produce higher similarity than clearly unrelated pairs
2. Model/input dimensions are consistent
3. Contrastive loss can be computed without errors
4. Existing embedding functionality remains unchanged
"""
import pytest
import torch
import torch.nn.functional as F

from embeddings.encoder import MockEmbeddingEncoder, SentenceTransformerEncoder
from ml.contrastive_model import (
    ContrastiveLoss,
    MemoryContrastiveNetwork,
    evaluate_pair_similarity,
    project_embedding,
    train_contrastive_step,
)


def test_contrastive_model_dimensions():
    """Verify input/output dimensions and normalization of MemoryContrastiveNetwork."""
    input_dim = 384
    hidden_dim = 256
    projection_dim = 128
    batch_size = 4

    model = MemoryContrastiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        dropout=0.0,
    )

    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    # Output shape must be [batch_size, projection_dim]
    assert output.shape == (batch_size, projection_dim)

    # Output vectors must be L2-normalized (length 1.0)
    norms = torch.norm(output, p=2, dim=-1)
    for norm_val in norms:
        assert abs(float(norm_val.item()) - 1.0) < 1e-4

    # Single vector input helper
    single_proj = project_embedding(model, dummy_input[0])
    assert single_proj.shape == (1, projection_dim)
    assert abs(float(torch.norm(single_proj, p=2, dim=-1).item()) - 1.0) < 1e-4


def test_contrastive_loss_computation():
    """Verify triplet margin contrastive loss computation without errors."""
    criterion = ContrastiveLoss(margin=0.5, temperature=0.07)
    proj_dim = 128
    batch_size = 3

    # Anchor vectors
    anchor = F.normalize(torch.randn(batch_size, proj_dim), p=2, dim=-1)

    # Positive vectors (close to anchor: small noise)
    positive = F.normalize(anchor + 0.05 * torch.randn(batch_size, proj_dim), p=2, dim=-1)

    # Negative vectors (orthogonal / far from anchor)
    negative = F.normalize(torch.randn(batch_size, proj_dim), p=2, dim=-1)

    loss = criterion(anchor, positive, negative)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0

    # If positive is much closer than negative, loss should be smaller than if positive is far
    far_pos = F.normalize(-anchor, p=2, dim=-1)
    close_neg = F.normalize(anchor + 0.01 * torch.randn(batch_size, proj_dim), p=2, dim=-1)
    high_loss = criterion(anchor, far_pos, close_neg)
    assert float(high_loss.item()) >= float(loss.item())


def test_contrastive_training_step():
    """Verify train_contrastive_step executes optimization and updates weights."""
    model = MemoryContrastiveNetwork(input_dim=384, projection_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss(margin=0.5)

    batch_size = 4
    anchor = torch.randn(batch_size, 384)
    # Configure pairs with margin violation to produce positive loss and active gradients
    pos = -anchor
    neg = anchor + 0.01 * torch.randn(batch_size, 384)

    # Get initial parameter values
    param_before = [p.clone() for p in model.parameters()]

    losses = []
    for _ in range(5):
        loss_val = train_contrastive_step(model, optimizer, anchor, pos, neg, criterion)
        losses.append(loss_val)

    assert len(losses) == 5
    assert all(isinstance(lv, float) for lv in losses)

    # Verify parameters changed
    params_changed = any(
        not torch.equal(p_b, p_a)
        for p_b, p_a in zip(param_before, model.parameters())
    )
    assert params_changed is True


def test_similar_pairs_higher_similarity():
    """Verify that similar memory pairs produce higher similarity scores than unrelated pairs."""
    encoder = MockEmbeddingEncoder(dim=384)

    # Similar pair: same topic (Python web development)
    text_anchor = "User develops backend microservices in Python with FastAPI."
    text_similar = "Python and FastAPI are the preferred tech stack for building APIs."

    # Dissimilar / unrelated item
    text_unrelated = "I had green tea with lemon for breakfast today."

    emb_anchor = encoder.encode_text(text_anchor)
    emb_similar = encoder.encode_text(text_similar)
    emb_unrelated = encoder.encode_text(text_unrelated)

    model = MemoryContrastiveNetwork(input_dim=384, hidden_dim=256, projection_dim=128, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = ContrastiveLoss(margin=0.5)

    # Fine-tune a few steps on this semantic relation
    t_anchor = torch.tensor([emb_anchor], dtype=torch.float32)
    t_similar = torch.tensor([emb_similar], dtype=torch.float32)
    t_unrelated = torch.tensor([emb_unrelated], dtype=torch.float32)

    for _ in range(10):
        train_contrastive_step(model, optimizer, t_anchor, t_similar, t_unrelated, criterion)

    # Evaluate similarity in contrastive projection space
    sim_similar = evaluate_pair_similarity(model, emb_anchor, emb_similar)
    sim_unrelated = evaluate_pair_similarity(model, emb_anchor, emb_unrelated)

    assert sim_similar > sim_unrelated, f"Expected {sim_similar} > {sim_unrelated}"


def test_existing_embedding_functionality_unchanged():
    """Verify that existing BaseEmbeddingEncoder implementations remain completely unchanged."""
    mock_enc = MockEmbeddingEncoder(dim=384)
    text = "Machine learning memory retention."
    vec = mock_enc.encode_text(text)

    # Dimensionality, typing, and deterministic nature preserved
    assert len(vec) == 384
    assert isinstance(vec[0], float)
    assert mock_enc.dimension == 384

    # Determinism: same input produces identical vector
    vec2 = mock_enc.encode_text(text)
    assert vec == vec2

    # Batch encoding preserved
    batch_vecs = mock_enc.encode_batch([text, "Second statement"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
    assert len(batch_vecs[1]) == 384

"""
Sentence Transformer Embedding Encoder with Environment Variable Configuration.
"""
from abc import ABC, abstractmethod
import hashlib
import logging
import os
from typing import List, Optional
import numpy as np
from core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class BaseEmbeddingEncoder(ABC):
    """Abstract interface for generating dense semantic vector embeddings."""

    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """Convert a single string into a normalized dense vector."""
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert a batch of strings into normalized dense vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by this encoder."""
        pass


class MockEmbeddingEncoder(BaseEmbeddingEncoder):
    """
    Deterministic mock encoder for lightweight offline development and testing.
    Generates unit-normalized 384-dimensional vectors based on text SHA256 hash.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot encode empty or whitespace text")
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [self.encode_text(t) for t in texts]


class SentenceTransformerEncoder(BaseEmbeddingEncoder):
    """
    Production encoder using HuggingFace Sentence Transformers (e.g. all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5).
    Reads configuration from environment variables if not explicitly provided.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> None:
        """Lazy load sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model: '%s' on device: '%s'", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = int(self._model.get_sentence_embedding_dimension())
            except ImportError as ie:
                logger.warning("sentence-transformers not installed. Falling back to MockEmbeddingEncoder.")
                raise EmbeddingError(f"sentence-transformers package not installed: {str(ie)}") from ie
            except Exception as e:
                logger.error("Failed to load SentenceTransformer '%s': %s", self.model_name, str(e))
                raise EmbeddingError(f"Embedding model loading failed: {str(e)}") from e

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def encode_text(self, text: str) -> List[float]:
        """Encode a single text into a normalized embedding vector."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot encode empty or whitespace text")
        self._load_model()
        try:
            emb = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return emb.tolist()
        except Exception as e:
            logger.error("Encoding failed for text: %s", str(e))
            raise EmbeddingError(f"Text encoding error: {str(e)}") from e

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into normalized embedding vectors."""
        if not texts:
            return []
        self._load_model()
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error("Batch encoding failed: %s", str(e))
            raise EmbeddingError(f"Batch encoding error: {str(e)}") from e


def get_embedding_encoder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> BaseEmbeddingEncoder:
    """
    Factory to instantiate the configured embedding encoder.
    Reads environment variables if arguments are omitted.
    """
    resolved_model = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    if resolved_model.lower() in ("mock", "test", "dummy"):
        return MockEmbeddingEncoder(dim=384)
    return SentenceTransformerEncoder(model_name=resolved_model, device=device, batch_size=batch_size)

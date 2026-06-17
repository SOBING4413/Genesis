from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
from loguru import logger
from app.config import get_settings


class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        if self._model is None:
            settings = get_settings()
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info("Embedding model loaded")
        return self

    def embed_text(self, text: str) -> List[float]:
        if self._model is None:
            raise RuntimeError("Model not initialized")
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("Model not initialized")
        embeddings = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [emb.tolist() for emb in embeddings]

    def compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        a = np.array(emb1)
        b = np.array(emb2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def compute_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        matrix = np.array(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / norms
        return np.dot(normalized, normalized.T)


embedding_service = EmbeddingService()
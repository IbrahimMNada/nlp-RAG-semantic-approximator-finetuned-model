"""
Reranker service using sentence-transformers CrossEncoder to re-score retrieval candidates.
"""
import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

from ....core.config import get_settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Re-scores candidate paragraphs using a CrossEncoder model."""

    def __init__(self):
        self._model: Optional[CrossEncoder] = None

    def _ensure_model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model on first use."""
        if self._model is None:
            settings = get_settings()
            logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
            self._model = CrossEncoder(settings.RERANKER_MODEL)
            logger.info("Reranker model loaded")
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int,
        content_key: str = "content",
    ) -> List[dict]:
        """
        Re-score and re-sort candidates using the cross-encoder.

        Args:
            query: The original search query text.
            candidates: List of candidate dicts from pgvector retrieval.
            top_k: Number of top results to return after reranking.
            content_key: Key in candidate dict that holds the text to compare.

        Returns:
            Top-k candidates sorted by reranker score (descending).
        """
        if not candidates:
            return candidates

        model = self._ensure_model()
        pairs = [(query, c[content_key]) for c in candidates]
        scores = model.predict(pairs).tolist()

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for candidate, score in scored[:top_k]:
            candidate["reranker_score"] = round(float(score), 4)
            results.append(candidate)

        logger.info(
            f"Reranked {len(candidates)} candidates → top {len(results)} "
            f"(best={results[0]['reranker_score']:.4f})"
        )

        return results

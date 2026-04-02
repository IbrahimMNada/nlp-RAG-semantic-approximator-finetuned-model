"""
Reranker service using Ollama's rerank API to re-score retrieval candidates.
"""
import logging
from typing import List

import httpx

from ....core.config import get_settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Re-scores candidate paragraphs using a reranker model via Ollama."""

    async def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int,
        content_key: str = "content",
    ) -> List[dict]:
        """
        Re-score and re-sort candidates via Ollama's rerank endpoint.

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

        settings = get_settings()
        documents = [c[content_key] for c in candidates]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_URL}/api/rerank",
                json={
                    "model": settings.RERANKER_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_k": top_k,
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", []):
            idx = item["index"]
            candidate = candidates[idx]
            candidate["reranker_score"] = round(item["relevance_score"], 4)
            results.append(candidate)

        logger.info(
            f"Reranked {len(candidates)} candidates → top {len(results)} "
            f"(best={results[0]['reranker_score']:.4f})"
        )

        return results

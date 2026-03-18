"""
Repository for Article database operations.
"""
import logging
from typing import List, Optional, Tuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ....core.config import get_settings
from ..entities import Article, ArticleParagraph, ArticleEmbedding1024, ParagraphEmbedding1024

logger = logging.getLogger(__name__)


class ArticleRepository:
    """Handles all Article-related database operations."""
    
    async def get_by_url(self, session: AsyncSession, url: str) -> Optional[Article]:
        """Get article by URL."""
        stmt = select(Article).where(Article.url == url)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_id(
        self, 
        session: AsyncSession, 
        article_id: int,
        include_paragraphs: bool = False
    ) -> Optional[Article]:
        """Get article by ID with optional paragraph loading."""
        stmt = select(Article).where(Article.id == article_id)
        if include_paragraphs:
            stmt = stmt.options(selectinload(Article.paragraphs))
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def upsert_article(
        self, 
        session: AsyncSession, 
        url: str, 
        scraped_data: dict
    ) -> Article:
        """
        Insert or update an article with its SEO data.
        
        Returns:
            The Article entity (flushed but not committed)
        """
        existing = await self.get_by_url(session, url)
        
        # Extract SEO data
        seo = scraped_data.get("seo", {})
        meta = seo.get("meta", {})
        
        article_data = {
            "title": scraped_data.get("title"),
            "author": scraped_data.get("author"),
            "last_update": scraped_data.get("last_update"),
            "seo_meta_description": meta.get("description"),
            "seo_meta_keywords": meta.get("keywords"),
            "seo_meta_viewport": meta.get("viewport"),
            "seo_meta_charset": meta.get("charset"),
            "seo_meta_theme_color": meta.get("theme-color"),
            "seo_meta_application_name": meta.get("application-name"),
            "seo_meta_thumbnail": meta.get("thumbnail"),
            "seo_title_tag": seo.get("title_tag"),
            "seo_canonical": seo.get("canonical"),
            "seo_headings": seo.get("headings"),
            "seo_open_graph": seo.get("open_graph"),
            "seo_twitter_cards": seo.get("twitter_cards"),
        }
        
        if existing:
            for key, value in article_data.items():
                setattr(existing, key, value)
            article = existing
            await session.flush()
            logger.info(f"Updating existing article: {url}")
        else:
            article = Article(url=url, **article_data)
            session.add(article)
            await session.flush()
            logger.info(f"Creating new article: {url}")
        
        return article
    
    async def delete_paragraphs(self, session: AsyncSession, article_id: int) -> None:
        """Delete all paragraphs for an article."""
        # Delete embeddings first (FK constraint)
        await session.execute(
            text("DELETE FROM paragraph_embeddings_1024 WHERE article_id = :article_id"),
            {"article_id": article_id}
        )
        await session.execute(
            text("DELETE FROM article_paragraphs WHERE article_id = :article_id"),
            {"article_id": article_id}
        )
    
    async def save_paragraphs(
        self, 
        session: AsyncSession, 
        article_id: int, 
        paragraph_texts: List[str]
    ) -> List[Tuple[int, str]]:
        """
        Save paragraphs for an article, filtering unwanted content.
        
        Returns:
            List of (paragraph_id, content) tuples
        """
        settings = get_settings()
        skip_patterns = settings.SKIP_PARAGRAPHS_CONTAINING
        
        for idx, paragraph_text in enumerate(paragraph_texts):
            # Skip unwanted paragraphs based on config
            should_skip = any(pattern in paragraph_text for pattern in skip_patterns)
            if should_skip:
                continue
            
            paragraph = ArticleParagraph(
                article_id=article_id,
                content=paragraph_text,
                order_index=idx
            )
            session.add(paragraph)
        
        await session.flush()
        
        # Query back to get IDs
        result = await session.execute(
            select(ArticleParagraph)
            .where(ArticleParagraph.article_id == article_id)
            .order_by(ArticleParagraph.order_index)
        )
        paragraphs = result.scalars().all()
        
        # Update article word_count and paragraph_count
        total_words = sum(len(p.content.split()) for p in paragraphs)
        article_result = await session.execute(
            select(Article).where(Article.id == article_id)
        )
        article = article_result.scalar_one_or_none()
        if article:
            article.word_count = total_words
            article.paragraph_count = len(paragraphs)
            await session.flush()
        
        return [(p.id, p.content) for p in paragraphs]
    
    async def get_all_articles(self, session: AsyncSession) -> List[Article]:
        """Get all articles."""
        result = await session.execute(select(Article))
        return list(result.scalars().all())
    
    async def get_all_articles_with_paragraphs(self, session: AsyncSession) -> List[Article]:
        """Get all articles with paragraphs eagerly loaded."""
        result = await session.execute(
            select(Article)
            .options(selectinload(Article.paragraphs))
            .order_by(Article.id)
        )
        return list(result.scalars().all())
    
    async def get_articles_without_embeddings(self, session: AsyncSession) -> List[Article]:
        """Get articles that don't have article-level embeddings."""
        query = select(Article).outerjoin(
            ArticleEmbedding1024,
            ArticleEmbedding1024.article_id == Article.id
        ).where(ArticleEmbedding1024.article_id.is_(None))
        
        result = await session.execute(query)
        return list(result.scalars().all())
    
    async def get_paragraph_embeddings(
        self, 
        session: AsyncSession, 
        article_id: int
    ) -> List[ParagraphEmbedding1024]:
        """Get all paragraph embeddings for an article."""
        result = await session.execute(
            select(ParagraphEmbedding1024)
            .where(ParagraphEmbedding1024.article_id == article_id)
        )
        return list(result.scalars().all())
    
    async def search_similar_articles(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        exclude_url: str,
        limit: int,
        threshold: float
    ) -> List[dict]:
        """
        Search for similar articles using vector similarity.
        
        Returns:
            List of dicts with article_id, title, url, similarity_score, paragraphs
        """
        query = text("""
            SELECT 
                a.id,
                a.title,
                a.url,
                1 - (ae.embedding <=> :query_embedding) as similarity_score
            FROM articles a
            JOIN article_embedding_1024 ae ON a.id = ae.article_id
            WHERE a.url != :query_url
            ORDER BY ae.embedding <=> :query_embedding
            LIMIT :limit
        """)
        
        result = await session.execute(
            query,
            {
                "query_embedding": str(query_embedding),
                "query_url": exclude_url,
                "limit": limit
            }
        )
        
        # Collect article IDs for batch paragraph fetch
        rows = list(result)
        if not rows:
            return []
        
        article_ids = [row.id for row in rows]
        
        # Batch fetch all paragraphs (fixes N+1)
        para_result = await session.execute(
            select(ArticleParagraph)
            .where(ArticleParagraph.article_id.in_(article_ids))
            .order_by(ArticleParagraph.article_id, ArticleParagraph.order_index)
        )
        all_paragraphs = para_result.scalars().all()
        
        # Group paragraphs by article_id
        paragraphs_by_article = {}
        for p in all_paragraphs:
            if p.article_id not in paragraphs_by_article:
                paragraphs_by_article[p.article_id] = []
            paragraphs_by_article[p.article_id].append(p.content)
        
        # Build results
        results = []
        for row in rows:
            similarity_score = float(row.similarity_score)
            if similarity_score < threshold:
                continue
            
            results.append({
                "article_id": row.id,
                "title": row.title,
                "url": row.url,
                "similarity_score": similarity_score,
                "paragraphs": paragraphs_by_article.get(row.id, [])
            })
        
        return results
    
    async def get_random_articles(
        self,
        session: AsyncSession,
        limit: int = 10
    ) -> List[Article]:
        """Get random articles from database."""
        query = text("""
            SELECT * FROM articles
            ORDER BY RANDOM()
            LIMIT :limit
        """)
        
        result = await session.execute(query, {"limit": limit})
        rows = result.fetchall()
        
        # Convert rows to Article objects
        articles = []
        for row in rows:
            article = Article(
                id=row.id,
                title=row.title,
                url=row.url,
                author=row.author,
                seo_meta_description=row.seo_meta_description,
                seo_meta_keywords=row.seo_meta_keywords,
                seo_title_tag=row.seo_title_tag,
                seo_canonical=row.seo_canonical,
                seo_meta_thumbnail=row.seo_meta_thumbnail
            )
            articles.append(article)
        
        return articles
    
    async def search_similar_paragraphs(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        min_words: int = 10
    ) -> List[dict]:
        """
        Search for similar paragraphs using vector similarity.
        
        Args:
            session: Database session
            query_embedding: Query embedding vector
            limit: Maximum number of paragraphs to return
            threshold: Minimum similarity score
            min_words: Minimum word count for paragraphs
            
        Returns:
            List of dicts with paragraph info and similarity scores
        """
        query = text("""
            SELECT 
                p.id as paragraph_id,
                p.article_id,
                p.content,
                p.order_index,
                a.title as article_title,
                a.url as article_url,
                1 - (pe.embedding <=> :query_embedding) as similarity_score,
                array_length(string_to_array(p.content, ' '), 1) as word_count
            FROM article_paragraphs p
            JOIN paragraph_embeddings_1024 pe ON p.id = pe.paragraph_id
            JOIN articles a ON p.article_id = a.id
            WHERE 
                     ( array_length(string_to_array(p.content, ' '), 1) >= :min_words) AND
                      1 - (pe.embedding <=> :query_embedding) >= :threshold
                
            ORDER BY pe.embedding <=> :query_embedding
            LIMIT :limit
        """)
        
        result = await session.execute(
            query,
            {
                "query_embedding": str(query_embedding),
                "limit": limit,
                "threshold": threshold,
                "min_words": min_words
            }
        )
        
        results = []
        for row in result:
            results.append({
                "paragraph_id": row.paragraph_id,
                "article_id": row.article_id,
                "article_title": row.article_title,
                "article_url": row.article_url,
                "content": row.content,
                "similarity_score": float(row.similarity_score),
                "order_index": row.order_index
            })
        
        return results


# Singleton
_article_repository_instance: Optional[ArticleRepository] = None


def get_article_repository() -> ArticleRepository:
    """Get or create ArticleRepository singleton."""
    global _article_repository_instance
    if _article_repository_instance is None:
        _article_repository_instance = ArticleRepository()
    return _article_repository_instance

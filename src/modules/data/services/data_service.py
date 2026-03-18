"""
Orchestration service for data processing operations.
Delegates to specialized services for specific concerns.
"""
import json
import logging
import time
from typing import Callable, AsyncContextManager
from urllib.parse import unquote

import aio_pika
import numpy as np
from sqlalchemy.orm import selectinload
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from  ....contracts.data import SimilarParagraph, SearchSimilarParagraphsRequestDto , SearchSimilarParagraphsResponseDto
from ..dtos.requests import ProcessFileDto, SearchSimilarDto
from ..dtos.responses import (
    ProcessFileResponseDto, 
    SearchSimilarResponseDto, 
    SimilarArticle,
    RandomArticlesResponseDto,
    RandomArticleDto,
)
from ..entities import Article
from ....core.database import get_async_db_session
from ....core.base_dtos import ResponseDto
from ....core.config import get_settings
from ....core.cache_service import cache_service
from ....abstractions.interfaces.web_scraper_interface import IWebScraper
from .web_scraper_factory import WebScraperFactory
from .embedding_service import EmbeddingService
from .article_repository import ArticleRepository

logger = logging.getLogger(__name__)

# Type alias for session factory
SessionFactory = Callable[[], AsyncContextManager[AsyncSession]]


class DataService:
    """Orchestration service for data processing operations."""
    
    def __init__(
        self, 
        scraper_factory: WebScraperFactory,
        embedding_service: EmbeddingService,
        article_repository: ArticleRepository,
        session_factory: SessionFactory = get_async_db_session,
    ):
        """
        Initialize the DataService.
        
        Args:
            scraper_factory: Factory that resolves scrapers by URL domain
            embedding_service: EmbeddingService instance
            article_repository: ArticleRepository instance
            session_factory: Factory for creating async DB sessions (injectable for testing)
        """
        self._scraper_factory = scraper_factory
        self._embedding_service = embedding_service
        self._article_repository = article_repository
        self._session_factory = session_factory
    
    
    async def process_url(self, process_dto: ProcessFileDto) -> ResponseDto[ProcessFileResponseDto]:
        """
        Process a URL by scraping its content and saving to database.
        
        Args:
            process_dto: DTO containing URL and processing parameters
            
        Returns:
            ResponseDto[ProcessFileResponseDto] with scraped data
        """
        try:
            url = unquote(str(process_dto.url))
            scraper = self._scraper_factory.get_scraper(url)
            scraped_data = await scraper.scrape_url(url)
            
            if not scraped_data or not scraped_data.get("title"):
                return ResponseDto[ProcessFileResponseDto].fail(1001, "urlNotValid")
            
            article = await self._save_to_database(url, scraped_data)
            response_data = self._build_response_from_article(article)
            
            return ResponseDto[ProcessFileResponseDto].success(response_data)
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
            return ResponseDto[ProcessFileResponseDto].fail(1002, f"Error processing URL: {str(e)}")
    
    async def _save_to_database(self, url: str, scraped_data: dict) -> Article:
        """Save or update article data in database."""
        async with self._session_factory() as session:
            # Check if exists to delete old paragraphs
            existing = await self._article_repository.get_by_url(session, url)
            if existing:
                await self._article_repository.delete_paragraphs(session, existing.id)
            
            # Upsert article
            article = await self._article_repository.upsert_article(session, url, scraped_data)
            
            # Save paragraphs
            paragraph_texts = scraped_data.get("article_text_paragraphs", [])
            paragraph_data = await self._article_repository.save_paragraphs(
                session, article.id, paragraph_texts
            )
            
            logger.info(f"Saved {len(paragraph_data)} paragraphs for article {article.id}")
            
            # Invalidate cache before embedding to avoid stale results on embedding failure
            await cache_service.invalidate_url(url)
            
            # Process embeddings
            settings = get_settings()
            if settings.QUEUE_ENABLED:
                # Commit transaction BEFORE queueing to avoid race condition
                await session.commit()
                logger.info(f"Committed article {article.id} to database")
                
                # Now queue the message - consumer will find the committed data
                await self._queue_embeddings_generation(article.id, paragraph_data)
                logger.info(f"Queued embeddings generation for article {article.id}")
            else:
                await self._embedding_service.generate_and_save_all(
                    session, article.id, paragraph_data
                )
            
            # Reload with paragraphs for response
            stmt = select(Article).where(Article.id == article.id).options(
                selectinload(Article.paragraphs)
            )
            result = await session.execute(stmt)
            article = result.scalar_one()
            session.expunge(article)
            
            return article
    
    async def generate_embeddings(self, session, article_id: int, paragraph_data: list) -> None:
        """
        Generate embeddings for paragraphs - delegates to EmbeddingService.
        Kept for backward compatibility with consumer.
        """
        await self._embedding_service.generate_and_save_all(session, article_id, paragraph_data)
    
    async def _queue_embeddings_generation(self, article_id: int, paragraph_data: list) -> None:
        """Queue embeddings generation task to RabbitMQ."""
        try:
            settings = get_settings()
            
            message_payload = {
                "article_id": article_id,
                "paragraphs": [
                    {"paragraph_id": p_id, "content": content}
                    for p_id, content in paragraph_data
                ]
            }
            
            connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
            async with connection:
                channel = await connection.channel()
                
                await channel.declare_queue(
                    settings.RABBITMQ_QUEUE_NAME,
                    durable=True
                )
                
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(message_payload).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                    ),
                    routing_key=settings.RABBITMQ_QUEUE_NAME
                )
                
                logger.info(f"Queued embeddings for article {article_id}")
                
        except Exception as e:
            logger.error(f"Failed to queue embeddings: {str(e)}", exc_info=True)
            raise
    
    def _build_response_from_article(self, article: Article) -> ProcessFileResponseDto:
        """Build response DTO from database article entity."""
        seo_data = {
            "meta": {
                "description": article.seo_meta_description,
                "keywords": article.seo_meta_keywords,
                "viewport": article.seo_meta_viewport,
                "charset": article.seo_meta_charset,
                "theme-color": article.seo_meta_theme_color,
                "application-name": article.seo_meta_application_name,
                "thumbnail": article.seo_meta_thumbnail
            },
            "title_tag": article.seo_title_tag,
            "canonical": article.seo_canonical,
            "headings": article.seo_headings,
            "open_graph": article.seo_open_graph,
            "twitter_cards": article.seo_twitter_cards
        }
        
        article_paragraphs = [
            p.content for p in sorted(article.paragraphs, key=lambda x: x.order_index)
        ]
        
        scraped_data = {
            "title": article.title,
            "author": article.author,
            "last_update": article.last_update,
            "article_text_paragraphs": article_paragraphs,
            "seo": seo_data
        }
        
        return ProcessFileResponseDto(
            scraped_data=scraped_data,
            title=article.title,
            author=article.author,
            last_update=article.last_update,
            seo=seo_data
        )
    
    async def process_file_content(self, content: str, file_type: str) -> ResponseDto[ProcessFileResponseDto]:
        """Process file content directly (not implemented)."""
        return ResponseDto[ProcessFileResponseDto].fail(
            1003, 
            f"File type {file_type} processing not implemented yet"
        )
    
    async def rebuild_index(self) -> ResponseDto[dict]:
        """Rebuild the HNSW vector indexes."""
        try:
            async with self._session_factory() as session:
                from sqlalchemy import text
                
                await session.execute(text("DROP INDEX IF EXISTS paragraph_embeddings_1024_embedding_idx"))
                await session.execute(text("""
                    CREATE INDEX paragraph_embeddings_1024_embedding_idx
                    ON paragraph_embeddings_1024 USING hnsw (embedding vector_cosine_ops)
                    WITH (m=16, ef_construction=64)
                """))
                
                await session.execute(text("DROP INDEX IF EXISTS article_embedding_1024_idx"))
                await session.execute(text("""
                    CREATE INDEX article_embedding_1024_idx
                    ON article_embedding_1024 USING hnsw (embedding vector_cosine_ops)
                    WITH (m=16, ef_construction=64)
                """))
                
                await session.commit()
                
                deleted_count = await cache_service.clear_all_similarity_cache()
                logger.info(f"Cleared {deleted_count} cache entries after index rebuild")
                
                return ResponseDto[dict].success({
                    "status": "Indexes rebuilt successfully",
                    "indexes_rebuilt": ["paragraph_embeddings_1024", "article_embedding_1024"],
                    "cache_cleared": deleted_count
                })
                
        except Exception as e:
            logger.error(f"Failed to rebuild indexes: {str(e)}", exc_info=True)
            return ResponseDto[dict].fail(1007, f"Index rebuild failed: {str(e)}")
    
    async def search_similar(self, search_dto: SearchSimilarDto) -> ResponseDto[SearchSimilarResponseDto]:
        """Search for similar articles based on URL content with Redis caching."""
        try:
            decoded_url = unquote(str(search_dto.url))
            scraper = self._scraper_factory.get_scraper(decoded_url)
            threshold = getattr(search_dto, 'threshold', 0.0)
            
            # Check cache first
            cached_results = await cache_service.get_cached_results(
                url=decoded_url,
                requested_limit=search_dto.limit,
                threshold=threshold
            )
            
            if cached_results is not None:
                scraped_data = await scraper.scrape_url(decoded_url)
                query_paragraphs = scraped_data.get("article_text_paragraphs", [])
                
                similar_articles = [
                    SimilarArticle(
                        article_id=r['article_id'],
                        title=r['title'],
                        url=r['url'],
                        similarity_score=r['similarity_score'],
                        paragraphs=r.get('paragraphs', [])
                    )
                    for r in cached_results
                ]
                
                return ResponseDto[SearchSimilarResponseDto].success(
                    SearchSimilarResponseDto(
                        query_url=decoded_url,
                        query_paragraphs=query_paragraphs,
                        similar_articles=similar_articles
                    )
                )
            
            # Cache miss - fetch from database
            scraped_data = await scraper.scrape_url(decoded_url)
            
            if not scraped_data or not scraped_data.get("article_text_paragraphs"):
                return ResponseDto[SearchSimilarResponseDto].fail(
                    1008, "Could not extract content from URL"
                )
            
            paragraphs = scraped_data.get("article_text_paragraphs", [])
            
            # Generate query embedding
            paragraph_embeddings = await self._embedding_service.generate_embeddings_batch(paragraphs)
            
            if not paragraph_embeddings:
                return ResponseDto[SearchSimilarResponseDto].fail(
                    1008, "Could not generate embeddings for content"
                )
            
            query_embedding = np.mean(paragraph_embeddings, axis=0).tolist()
            logger.info(f"Generated query embedding from {len(paragraphs)} paragraphs")
            
            # Search
            query_start = time.time()
            async with self._session_factory() as session:
                results = await self._article_repository.search_similar_articles(
                    session,
                    query_embedding=query_embedding,
                    exclude_url=decoded_url,
                    limit=search_dto.limit,
                    threshold=search_dto.threshold
                )
            
            query_duration = time.time() - query_start
            logger.info(f"Similarity search completed in {query_duration:.2f}s")
            
            similar_articles = [
                SimilarArticle(
                    article_id=r['article_id'],
                    title=r['title'],
                    url=r['url'],
                    similarity_score=r['similarity_score'],
                    paragraphs=r['paragraphs']
                )
                for r in results
            ]
            
            # Cache results
            settings = get_settings()
            await cache_service.set_cached_results(
                url=decoded_url,
                results=results,
                limit=search_dto.limit,
                threshold=search_dto.threshold,
                ttl=settings.CACHE_TTL
            )
            
            logger.info(f"Found {len(similar_articles)} similar articles for {decoded_url}")
            
            return ResponseDto[SearchSimilarResponseDto].success(
                SearchSimilarResponseDto(
                    query_url=decoded_url,
                    query_paragraphs=paragraphs,
                    similar_articles=similar_articles
                )
            )
            
        except Exception as e:
            logger.error(f"Error searching for similar articles: {str(e)}", exc_info=True)
            return ResponseDto[SearchSimilarResponseDto].fail(1009, f"Search failed: {str(e)}")
    
    async def compute_article_embeddings(self) -> ResponseDto[dict]:
        """Compute article-level embeddings by averaging paragraph embeddings."""
        try:
            async with self._session_factory() as session:
                articles = await self._article_repository.get_all_articles(session)
                
                logger.info(f"Computing article embeddings for {len(articles)} articles")
                
                processed = 0
                skipped = 0
                errors = 0
                
                for article in articles:
                    try:
                        embeddings = await self._article_repository.get_paragraph_embeddings(
                            session, article.id
                        )
                        
                        if not embeddings:
                            logger.warning(f"No paragraph embeddings for article {article.id}")
                            skipped += 1
                            continue
                        
                        vectors = [pe.embedding for pe in embeddings]
                        await self._embedding_service.save_article_embedding(
                            session, article.id, vectors
                        )
                        
                        processed += 1
                        
                        if processed % 100 == 0:
                            await session.commit()
                            logger.info(f"Processed {processed} articles...")
                        
                    except Exception as e:
                        logger.error(f"Error processing article {article.id}: {str(e)}")
                        errors += 1
                
                await session.commit()
                
                result = {
                    "total_articles": len(articles),
                    "processed": processed,
                    "skipped": skipped,
                    "errors": errors
                }
                
                logger.info(f"Article embedding computation complete: {result}")
                return ResponseDto[dict].success(result)
                
        except Exception as e:
            logger.error(f"Failed to compute article embeddings: {str(e)}", exc_info=True)
            return ResponseDto[dict].fail(1010, f"Computation failed: {str(e)}")
    
    async def process_articles_without_embeddings(self) -> ResponseDto[dict]:
        """Find and reprocess articles without embeddings."""
        try:
            async with self._session_factory() as session:
                articles = await self._article_repository.get_articles_without_embeddings(session)
                
                total = len(articles)
                logger.info(f"Found {total} articles without embeddings")
                
                if total == 0:
                    return ResponseDto[dict].success({
                        "total_found": 0,
                        "processed": 0,
                        "failed": 0,
                        "message": "No articles without embeddings found"
                    })
                
                processed = 0
                failed = 0
                
                for article in articles:
                    try:
                        logger.info(f"Reprocessing article {article.id}: {article.url}")
                        
                        response = await self.process_url(ProcessFileDto(url=article.url))
                        
                        if response.status_code == 200:
                            processed += 1
                        else:
                            failed += 1
                            logger.warning(f"Failed: {response.error_description}")
                        
                    except Exception as e:
                        failed += 1
                        logger.error(f"Error processing article {article.id}: {str(e)}")
                
                result = {
                    "total_found": total,
                    "processed": processed,
                    "failed": failed,
                    "message": f"Processed {processed} out of {total} articles"
                }
                
                logger.info(f"Completed: {result}")
                return ResponseDto[dict].success(result)
                
        except Exception as e:
            logger.error(f"Failed to process articles: {str(e)}", exc_info=True)
            return ResponseDto[dict].fail(1011, f"Processing failed: {str(e)}")
    
    async def search_similar_paragraphs(
        self, 
        search_dto: SearchSimilarParagraphsRequestDto
    ) -> ResponseDto[SearchSimilarParagraphsResponseDto]:
        """
        Search for similar paragraphs based on text input.
        
        This method is independent and focused on paragraph-level similarity,
        making it suitable for RAG modules to use without depending on URL scraping.
        
        Args:
            search_dto: DTO containing text query and search parameters
            
        Returns:
            ResponseDto containing similar paragraphs with their metadata
        """
        try:
            # Generate embedding for the query text
            query_embedding = await self._embedding_service.generate_embedding(search_dto.text)
            
            if not query_embedding:
                return ResponseDto[SearchSimilarParagraphsResponseDto].fail(
                    1012, "Could not generate embedding for query text"
                )
            
            logger.info(f"Generated query embedding for text: '{search_dto.text[:50]}...'")
            
            # Search for similar paragraphs
            query_start = time.time()
            async with self._session_factory() as session:
                results = await self._article_repository.search_similar_paragraphs(
                    session,
                    query_embedding=query_embedding,
                    limit=search_dto.limit,
                    threshold=search_dto.threshold,
                    min_words=search_dto.min_words
                )
            
            query_duration = time.time() - query_start
            logger.info(
                f"Paragraph search completed in {query_duration:.2f}s, "
                f"found {len(results)} matches (min_words={search_dto.min_words})"
            )
            
            # Build response
            similar_paragraphs = [
                SimilarParagraph(
                    paragraph_id=r['paragraph_id'],
                    article_id=r['article_id'],
                    article_title=r['article_title'],
                    article_url=r['article_url'],
                    content=r['content'],
                    similarity_score=r['similarity_score'],
                    order_index=r['order_index']
                )
                for r in results
            ]
            
            return ResponseDto[SearchSimilarParagraphsResponseDto].success(
                SearchSimilarParagraphsResponseDto(
                    query_text=search_dto.text,
                    similar_paragraphs=similar_paragraphs
                )
            )
            
        except Exception as e:
            logger.error(f"Error searching for similar paragraphs: {str(e)}", exc_info=True)
            return ResponseDto[SearchSimilarParagraphsResponseDto].fail(
                1013, f"Paragraph search failed: {str(e)}"
            )
    
    async def get_random_articles(self, limit: int = 10) -> ResponseDto[RandomArticlesResponseDto]:
        """
        Get random articles with their titles, URLs, and SEO metadata.
        
        Args:
            limit: Number of random articles to return (default: 10)
            
        Returns:
            ResponseDto containing list of random articles with SEO metadata
        """
        try:
            async with self._session_factory() as session:
                articles = await self._article_repository.get_random_articles(session, limit=limit)
                
                random_articles = [
                    RandomArticleDto(
                        id=article.id,
                        title=article.title,
                        url=article.url,
                        author=article.author,
                        seo_meta_description=article.seo_meta_description,
                        seo_meta_keywords=article.seo_meta_keywords,
                        seo_title_tag=article.seo_title_tag,
                        seo_canonical=article.seo_canonical,
                        seo_meta_thumbnail=article.seo_meta_thumbnail
                    )
                    for article in articles
                ]
                
                logger.info(f"Retrieved {len(random_articles)} random articles")
                
                return ResponseDto[RandomArticlesResponseDto].success(
                    RandomArticlesResponseDto(
                        articles=random_articles,
                        total_count=len(random_articles)
                    )
                )
                
        except Exception as e:
            logger.error(f"Error retrieving random articles: {str(e)}", exc_info=True)
            return ResponseDto[RandomArticlesResponseDto].fail(
                1014, f"Failed to retrieve random articles: {str(e)}"
            )

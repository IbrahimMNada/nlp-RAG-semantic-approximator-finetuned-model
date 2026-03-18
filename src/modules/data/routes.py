from fastapi import APIRouter, Request, Depends
from ...core.base_dtos import ResponseDto
from  ...contracts.data import SearchSimilarParagraphsRequestDto , SearchSimilarParagraphsResponseDto
from .dtos import (
    ProcessFileDto, 
    ProcessFileResponseDto, 
    SearchSimilarDto, 
    SearchSimilarResponseDto,
    RandomArticlesResponseDto,
)
from .dependencies import DataServiceDep
from ...core.security import limiter, verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/process", response_model=ResponseDto[ProcessFileResponseDto])
@limiter.limit("10/minute")
async def process_url(
    request: Request,
    process_file_dto: ProcessFileDto,
    data_service: DataServiceDep,
):
    """
    Process a URL by scraping its content and extracting structured data.
    
    Args:
        process_file_dto: DTO containing the URL to process and other parameters
        
    Returns:
        ResponseDto containing the scraped and processed data
    """
    # Use the service to process the URL
    return await data_service.process_url(process_file_dto)


@router.post("/search-similar", response_model=ResponseDto[SearchSimilarResponseDto])
@limiter.limit("30/minute")
async def search_similar(
    request: Request,
    search_dto: SearchSimilarDto,
    data_service: DataServiceDep,
):
    """
    Search for similar articles based on URL content using vector similarity.
    
    Args:
        search_dto: DTO containing the URL to search for and limit
        
    Returns:
        ResponseDto containing list of similar articles with similarity scores
    """
    return await data_service.search_similar(search_dto)


@router.post("/search-similar-paragraphs", response_model=ResponseDto[SearchSimilarParagraphsResponseDto], include_in_schema=True)
@limiter.limit("30/minute")
async def search_similar_paragraphs(
    request: Request,
    search_dto: SearchSimilarParagraphsRequestDto,
    data_service: DataServiceDep,
):
    """
    Search for similar paragraphs based on text input using vector similarity.
    
    This endpoint is optimized for RAG (Retrieval-Augmented Generation) use cases,
    returning individual relevant paragraphs rather than full articles.
    
    Args:
        search_dto: DTO containing the text query, limit, and similarity threshold
        
    Returns:
        ResponseDto containing list of similar paragraphs with metadata and similarity scores
    """
    return await data_service.search_similar_paragraphs(search_dto)


@router.post("/rebuild-index", response_model=ResponseDto[dict])
@limiter.limit("2/minute")
async def rebuild_index(request: Request, data_service: DataServiceDep):
    """
    Rebuild the HNSW vector index.
    
    Returns:
        ResponseDto with rebuild status
    """
    return await data_service.rebuild_index()


@router.post("/compute-article-embeddings", response_model=ResponseDto[dict])
async def compute_article_embeddings(data_service: DataServiceDep):
    """
    Compute article-level embeddings by averaging paragraph embeddings for all articles.
    This precomputes article embeddings to speed up similarity searches.
    
    Returns:
        ResponseDto with computation statistics (processed, skipped, errors)
    """
    return await data_service.compute_article_embeddings()


@router.post("/process-articles-without-embeddings", response_model=ResponseDto[dict])
async def process_articles_without_embeddings(data_service: DataServiceDep):
    """
    Find and reprocess all articles that don't have embeddings yet.
    This is useful for backfilling embeddings for existing articles.
    
    Returns:
        ResponseDto with processing statistics (total_found, processed, failed)
    """
    return await data_service.process_articles_without_embeddings()


@router.get("/random-articles", response_model=ResponseDto[RandomArticlesResponseDto])
async def get_random_articles(
    data_service: DataServiceDep,
    limit: int = 10
):
    """
    Get random articles with their titles, URLs, and SEO metadata.
    
    Args:
        limit: Number of random articles to return (default: 10)
        
    Returns:
        ResponseDto containing list of random articles with SEO metadata
    """
    return await data_service.get_random_articles(limit=limit)

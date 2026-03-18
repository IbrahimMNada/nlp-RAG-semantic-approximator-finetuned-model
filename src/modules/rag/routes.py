"""RAG module routes."""
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field
from typing import Optional
from ...core.base_dtos import ResponseDto
from ...contracts.data import SearchSimilarParagraphsRequestDto, SearchSimilarParagraphsResponseDto
from .dtos.responses import AnswerResponseDto
from .dependencies import RagServiceDep
from ...core.security import limiter, verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


class AskWithContextRequestDto(BaseModel):
    """Request DTO for ask-with-context endpoint."""
    question: str = Field(..., description="Question to ask ChatGPT (also used to search for context)")
    limit: Optional[int] = Field(3, description="Maximum number of context paragraphs", ge=1, le=10)
    similarity_threshold: Optional[float] = Field(0.5, description="Minimum similarity score", ge=0.0, le=1.0)
    temperature: Optional[float] = Field(0.7, description="ChatGPT temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens in response", gt=0)


@router.post("/search-context", response_model=ResponseDto[SearchSimilarParagraphsResponseDto])
@limiter.limit("30/minute")
async def search_context(
    request: Request,
    search_request: SearchSimilarParagraphsRequestDto,
    rag_service: RagServiceDep,
):
    """
    Search for similar paragraphs from the data module.
    
    This endpoint calls the data module's search-similar-paragraphs endpoint,
    demonstrating inter-module communication using the type-safe ModulesHttpClient.
    
    Args:
        search_request: Search request containing text query and parameters
        rag_service: Injected RAG service
        
    Returns:
        ResponseDto containing similar paragraphs with metadata
    """
    return await rag_service.search_context(search_request)


@router.post("/ask-with-context", response_model=ResponseDto[AnswerResponseDto])
@limiter.limit("20/minute")
async def ask_with_context(
    request: Request,
    ask_request: AskWithContextRequestDto,
    rag_service: RagServiceDep,
):
    """
    Search for context and ask ChatGPT a question based on that context.
    
    This endpoint combines context retrieval from the data module with ChatGPT
    to provide answers grounded in your data. The question is used both to search
    for relevant context and as the question to ask ChatGPT.
    
    Args:
        request: Request containing question and search parameters
        rag_service: Injected RAG service
        
    Returns:
        ResponseDto containing ChatGPT's answer with sources
    """
    return await rag_service.ask_with_context(
        question=ask_request.question,
        limit=ask_request.limit,
        similarity_threshold=ask_request.similarity_threshold,
        temperature=ask_request.temperature,
        max_tokens=ask_request.max_tokens,
    )

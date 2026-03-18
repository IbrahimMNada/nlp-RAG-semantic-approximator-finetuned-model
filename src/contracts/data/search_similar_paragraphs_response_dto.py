"""Response DTO for paragraph similarity search."""
from typing import List
from pydantic import BaseModel, Field


class SimilarParagraph(BaseModel):
    """A single similar paragraph result."""
    
    paragraph_id: int = Field(..., description="ID of the paragraph")
    article_id: int = Field(..., description="ID of the article containing this paragraph")
    article_title: str = Field(..., description="Title of the article")
    article_url: str = Field(..., description="URL of the article")
    content: str = Field(..., description="Paragraph content")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    order_index: int = Field(..., description="Position of paragraph in article")


class SearchSimilarParagraphsResponseDto(BaseModel):
    """Response for paragraph similarity search."""
    
    query_text: str = Field(..., description="The input text that was searched")
    similar_paragraphs: List[SimilarParagraph] = Field(
        default_factory=list, 
        description="List of similar paragraphs ordered by similarity"
    )

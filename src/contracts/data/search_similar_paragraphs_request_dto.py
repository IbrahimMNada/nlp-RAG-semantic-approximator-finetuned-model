"""DTO for searching similar paragraphs based on text input."""
from pydantic import BaseModel, Field


class SearchSimilarParagraphsRequestDto(BaseModel):
    """Request DTO for searching similar paragraphs."""
    
    text: str = Field(..., description="Text to find similar paragraphs for", min_length=1)
    limit: int = Field(default=10, description="Maximum number of paragraphs to return", ge=1, le=100)
    threshold: float = Field(default=0.0, description="Minimum similarity score (0-1)", ge=0.0, le=1.0)
    min_words: int = Field(default=10, description="Minimum word count for paragraphs", ge=0)

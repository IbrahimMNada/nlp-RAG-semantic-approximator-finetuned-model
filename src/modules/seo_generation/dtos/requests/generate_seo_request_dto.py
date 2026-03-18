"""Request DTO for SEO generation."""
from pydantic import BaseModel, Field


class GenerateSeoRequestDto(BaseModel):
    """Request DTO for generate SEO endpoint."""
    text: str = Field(..., description="Input text for SEO content generation", min_length=1)
    max_length: int = Field(512, description="Maximum length of generated text", ge=50, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Nucleus sampling parameter", ge=0.0, le=1.0)

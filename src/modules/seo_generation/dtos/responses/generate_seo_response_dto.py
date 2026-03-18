"""Response DTO for SEO generation."""
from pydantic import BaseModel, Field


class GenerateSeoResponseDto(BaseModel):
    """Response DTO for SEO generation."""
    generated_text: str = Field(..., description="Generated SEO content")
    input_text: str = Field(..., description="Original input text")
    model_name: str = Field(..., description="Name of the model used")

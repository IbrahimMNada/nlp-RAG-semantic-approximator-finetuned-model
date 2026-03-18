"""Answer response DTO."""
from pydantic import BaseModel, Field
from typing import Optional, List


class AnswerResponseDto(BaseModel):
    """Response DTO for answer endpoints."""
    message: str = Field(..., description="AI-generated response message")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    context_used: Optional[bool] = Field(False, description="Whether context was used")
    sources: Optional[List[str]] = Field(default_factory=list, description="Sources used for context")
    cost: float = Field(default=0.0, description="Cost of the API call in USD")

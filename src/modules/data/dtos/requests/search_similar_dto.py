from pydantic import BaseModel, Field, HttpUrl

class SearchSimilarDto(BaseModel):
    url: HttpUrl = Field(..., description="URL to find similar articles for")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of similar articles to return")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold (0.0 to 1.0)")

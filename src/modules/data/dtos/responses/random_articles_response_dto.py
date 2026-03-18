"""DTO for random articles response."""
from typing import Optional
from pydantic import BaseModel


class RandomArticleDto(BaseModel):
    """Single random article with basic info."""
    id: int
    title: str
    url: str
    author: Optional[str] = None
    seo_meta_description: Optional[str] = None
    seo_meta_keywords: Optional[str] = None
    seo_title_tag: Optional[str] = None
    seo_canonical: Optional[str] = None
    seo_meta_thumbnail: Optional[str] = None


class RandomArticlesResponseDto(BaseModel):
    """Response containing list of random articles."""
    articles: list[RandomArticleDto]
    total_count: int

from pydantic import BaseModel, Field
from typing import List

class SimilarArticle(BaseModel):
    #article_id: int
    title: str
    url: str
    similarity_score: float
    #paragraphs: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True

class SearchSimilarResponseDto(BaseModel):
    query_url: str
    #query_paragraphs: List[str] = Field(default_factory=list)
    similar_articles: List[SimilarArticle]
    
    class Config:
        from_attributes = True

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ProcessFileResponseDto(BaseModel):
    scraped_data: Optional[Dict[str, Any]] = None
    
    # Extracted data fields
    title: Optional[str] = None
    author: Optional[str] = None
    last_update: Optional[str] = None
    breadcrumbs: Optional[List[str]] = None
    summary: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = None
    references: Optional[List[str]] = None
    related_articles: Optional[List[Dict[str, str]]] = None
    all_links: Optional[List[Dict[str, str]]] = None
    all_images: Optional[List[Dict[str, Optional[str]]]] = None
    microdata: Optional[Dict[str, List[str]]] = None
    
    # SEO data
    seo: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True
"""
Abstract interface for web scraper implementations.
This allows multiple scraping strategies to be plugged in.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class IWebScraper(ABC):
    """Interface for web scraper implementations."""
    
    @abstractmethod
    async def scrape_url(self, url: str) -> Optional[Dict]:
        """
        Scrape content from a URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing scraped data with structure:
            {
                "title": str,
                "author": str,
                "last_update": str,
                "article_text_paragraphs": List[str],
                "seo": Dict
            }
        """
        pass
    
    @abstractmethod
    def get_scraper_name(self) -> str:
        """Return the name/identifier of this scraper implementation."""
        pass

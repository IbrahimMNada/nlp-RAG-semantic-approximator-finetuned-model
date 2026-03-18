"""
Factory for resolving the appropriate IWebScraper implementation based on URL domain.
"""
from typing import Dict, Optional, Type
from urllib.parse import urlparse

from ....abstractions.interfaces.web_scraper_interface import IWebScraper
from .web_scraper import DefaultWebScraper


class WebScraperFactory:
    """
    Resolves IWebScraper implementations by matching the URL domain
    against a registered domain-to-scraper mapping.
    Falls back to the DefaultWebScraper when no match is found.
    """

    def __init__(self) -> None:
        self._domain_map: Dict[str, IWebScraper] = {}
        self._default_scraper: IWebScraper = DefaultWebScraper()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, domain: str, scraper: IWebScraper) -> None:
        """
        Register a scraper instance for a specific domain.

        Args:
            domain: The domain to associate (e.g. "example.com").
            scraper: An IWebScraper implementation instance.
        """
        self._domain_map[domain.lower()] = scraper

    def set_default_scraper(self, scraper: IWebScraper) -> None:
        """Override the default fallback scraper."""
        self._default_scraper = scraper

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def get_scraper(self, url: str) -> IWebScraper:
        """
        Return the scraper registered for the URL's domain.
        Falls back to the default scraper if the domain is not registered.

        Args:
            url: The target URL.

        Returns:
            An IWebScraper instance appropriate for the URL's domain.
        """
        domain = self._extract_domain(url)
        if domain and domain in self._domain_map:
            return self._domain_map[domain]
        return self._default_scraper

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        """Extract the hostname (without www.) from the URL."""
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            if hostname.startswith("www."):
                hostname = hostname[4:]
            return hostname.lower() if hostname else None
        except Exception:
            return None

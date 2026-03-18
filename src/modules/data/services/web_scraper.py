import httpx
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
from ....abstractions.interfaces.web_scraper_interface import IWebScraper
from ....core.config import get_settings


class DefaultWebScraper(IWebScraper):
    """
    BeautifulSoup-based web scraper implementation.
    Fetches HTML content from URLs and extracts structured data.
    """
    
    def __init__(self, timeout: int = 30, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the WebScraper.
        
        Args:
            timeout: Request timeout in seconds
            headers: Optional custom headers for requests
        """
        self.timeout = timeout
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout, follow_redirects=True)
    
    def get_scraper_name(self) -> str:
        """Return the name of this scraper implementation."""
        return "defaultWebScraper"
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape data from a given URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing scraped data
            
        Raises:
            httpx.HTTPError: If the request fails
            Exception: If parsing fails
        """
        try:
            # Make GET request
            response = await self._make_request(url)
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract data
            data = self._extract_data(soup, url)
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to scrape URL {url}: {str(e)}")
    
    async def _make_request(self, url: str) -> httpx.Response:
        """
        Make HTTP GET request to the URL.
        
        Args:
            url: The URL to request
            
        Returns:
            Response object
        """
        response = await self.client.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        return response
    
    def _extract_data(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        """
        Extract structured data from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative links
            
        Returns:
            Dictionary containing extracted data
        """
        data = {}
        
        # ---------- Meta ----------
        data["title"] = self._extract_title(soup)
        data["author"] = self._extract_author(soup)
        data["last_update"] = self._extract_last_update(soup)
        # data["breadcrumbs"] = self._extract_breadcrumbs(soup)
        
        # # ---------- Summary ----------
        # data["summary"] = self._extract_summary(soup)
        
        # # ---------- Article Body ----------
        # data["sections"] = self._extract_sections(soup)
        
        # ---------- Article Text Paragraphs ----------
        data["article_text_paragraphs"] = self._extract_article_text_paragraphs(soup)
        
        # # ---------- References ----------
        # data["references"] = self._extract_references(soup)
        
        # # ---------- Related Articles ----------
        # data["related_articles"] = self._extract_related_articles(soup, base_url)
        
        # # ---------- All Links ----------
        # data["all_links"] = self._extract_all_links(soup, base_url)
        
        # # ---------- All Images ----------
        # data["all_images"] = self._extract_all_images(soup, base_url)
        
        # # ---------- Microdata (schema.org) ----------
        # data["microdata"] = self._extract_microdata(soup)
        
        # ---------- SEO Data ----------
        data["seo"] = self._extract_seo_data(soup, base_url)
        
        return data
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title."""
        title_elem = soup.select_one("h1.title")
        return title_elem.get_text(strip=True) if title_elem else None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information."""
        author_elem = soup.select_one(".article-author .info a")
        return author_elem.get_text(strip=True) if author_elem else None
    
    def _extract_last_update(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract last update date."""
        update_elem = soup.select_one(".article-author .info span[itemprop='dateModified']")
        return update_elem.get_text(strip=True) if update_elem else None
    
    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> List[str]:
        """Extract breadcrumb navigation."""
        return [b.get_text(strip=True) for b in soup.select("ul.breadcrumbs li span[itemprop='name']")]
    
    def _extract_summary(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article summary."""
        summary_elem = soup.select_one(".article-summary p")
        return summary_elem.get_text(strip=True) if summary_elem else None
    
    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract article body sections."""
        sections = []
        current_section = {"heading": "Intro", "content": []}
        
        article_body = soup.select(".article-body")
        if not article_body:
            return sections
        
        for el in article_body[0].children:
            if hasattr(el, 'name'):  # Check if element has a name attribute
                if el.name == "h2":
                    # Start new section
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"heading": el.get_text(" ", strip=True), "content": []}
                elif el.name in ["p", "ul", "ol", "table", "img"]:
                    if el.name != "img":
                        text = el.get_text(" ", strip=True)
                    else:
                        text = f"[Image: {el.get('src')}, alt={el.get('alt')}]"
                    
                    if text:
                        current_section["content"].append(text)
        
        # Add final section if it has content
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def _chunk_text(self, text: str, max_words: int = 110, overlap: int = 20) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            max_words: Maximum words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        
        if len(words) <= max_words:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + max_words
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            # Move start forward, accounting for overlap
            start = end - overlap
            
            # Prevent infinite loop if we're at the end
            if end >= len(words):
                break
        
        return chunks

    def _extract_article_text_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract all paragraphs and h2 headings under the DOM element with class 'article-text'.
        Concatenates all content into one paragraph, then chunks it into overlapping segments.
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            List of chunked text segments from the article
        """
        all_text_parts = []
        
        # Find the element with class 'article-text'
        article_text_element = soup.select_one(".article-text")
        
        if article_text_element:
            # Remove any elements with id="toc"
            toc_elements = article_text_element.find_all(id="toc")
            for toc in toc_elements:
                toc.decompose()
            
            # Iterate through all direct and nested children to maintain document order
            for element in article_text_element.descendants:
                # Check if the element is a p or h2 tag
                if hasattr(element, 'name') and element.name in ['p', 'h2']:
                    text = element.get_text(separator=' ', strip=True)
                    if text and len(text.split()) > 1:  # Ignore single-word items
                        all_text_parts.append(text)
                
                # Check if the element is a ul or ol tag (but not related-articles-list1 or references)
                elif hasattr(element, 'name') and element.name in ['ul', 'ol']:
                    # Skip if element has class 'related-articles-list1', 'references' or is inside one
                    if 'related-articles-list1' in element.get('class', []) or 'references' in element.get('class', []):
                        continue
                    if element.find_parent(class_='related-articles-list1') or element.find_parent(class_='references'):
                        continue
                    
                    list_items = element.find_all('li', recursive=False)
                    if list_items:
                        item_texts = [li.get_text(separator=' ',strip=True) for li in list_items if li.get_text(strip=True)]
                        if item_texts:
                            concatenated_text = ', '.join(item_texts)
                            all_text_parts.append(concatenated_text)

        

        # Filter out unwanted paragraphs based on config
        settings = get_settings()
        skip_patterns = settings.SKIP_PARAGRAPHS_CONTAINING
        
        filtered_text_parts = []
        for paragraph_text in all_text_parts:
            # Skip unwanted paragraphs based on config
            should_skip = any(pattern in paragraph_text for pattern in skip_patterns)
            if should_skip:
                continue
            filtered_text_parts.append(paragraph_text)
        
        # Concatenate all text parts into one paragraph
        full_text = ' '.join(filtered_text_parts)
        
        # Chunk the entire concatenated text
        content_elements = self._chunk_text(full_text) if full_text else []
        
        return content_elements

    def _extract_references(self, soup: BeautifulSoup) -> List[str]:
        """Extract references list."""
        return [ref.get_text(" ", strip=True) for ref in soup.select("ol.references li")]
    
    def _extract_related_articles(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract related articles."""
        related = []
        for a in soup.select("#related-list2 .categories-list a"):
            title = a.get_text(strip=True)
            url = self._resolve_url(a.get("href"), base_url)
            if title and url:
                related.append({"title": title, "url": url})
        return related
    
    def _extract_all_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            url = self._resolve_url(a.get("href"), base_url)
            if url:  # Only include if URL is valid
                links.append({"text": text, "url": url})
        return links
    
    def _extract_all_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Optional[str]]]:
        """Extract all images from the page."""
        images = []
        for img in soup.find_all("img"):
            src = self._resolve_url(img.get("src"), base_url)
            alt = img.get("alt")
            if src:  # Only include if src is valid
                images.append({"src": src, "alt": alt})
        return images
    
    def _extract_microdata(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract microdata (schema.org) properties."""
        microdata = {}
        for tag in soup.find_all(attrs={"itemprop": True}):
            prop = tag["itemprop"]
            value = tag.get_text(strip=True) or tag.get("content")
            if value:
                microdata.setdefault(prop, []).append(value)
        return microdata
    
    def _extract_seo_data(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        """Extract SEO-related data from the page."""
        seo_data = {}
        
        # Basic SEO elements
        seo_data["meta"] = self._extract_meta_tags(soup)
        seo_data["title_tag"] = self._extract_title_tag(soup)
        seo_data["headings"] = self._extract_headings(soup)
        seo_data["open_graph"] = self._extract_open_graph(soup, base_url)
        seo_data["twitter_cards"] = self._extract_twitter_cards(soup, base_url)
        seo_data["canonical"] = self._extract_canonical(soup, base_url)
      #  seo_data["robots"] = self._extract_robots_meta(soup)
      #  seo_data["json_ld"] = self._extract_json_ld(soup)
      #  seo_data["hreflang"] = self._extract_hreflang(soup, base_url)
      #  seo_data["icons"] = self._extract_icons(soup, base_url)
      #  seo_data["dns_prefetch"] = self._extract_dns_prefetch(soup)
        
        return seo_data
    
    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract standard meta tags."""
        meta_data = {}
        
        # Common meta tags
        meta_tags = {
            "description": ["name", "description"],
            "keywords": ["name", "keywords"],
            "author": ["name", "author"],
            "robots": ["name", "robots"],
            "viewport": ["name", "viewport"],
            "charset": ["charset", None],
            "generator": ["name", "generator"],
            "theme-color": ["name", "theme-color"],
            "application-name": ["name", "application-name"],
            "thumbnail": ["name", "thumbnail"],
            "revisit-after": ["name", "revisit-after"],
            "language": ["name", "language"],
            "distribution": ["name", "distribution"],
            "rating": ["name", "rating"],
            "copyright": ["name", "copyright"],
        }
        
        for key, (attr, value) in meta_tags.items():
            if value:
                meta_tag = soup.find("meta", attrs={attr: value})
            else:
                meta_tag = soup.find("meta", attrs={attr: True})
            
            if meta_tag:
                content = meta_tag.get("content") if value else meta_tag.get(attr)
                if content:
                    meta_data[key] = content
        
        # Also extract http-equiv meta tags
        http_equiv_tags = soup.find_all("meta", attrs={"http-equiv": True})
        for tag in http_equiv_tags:
            http_equiv = tag.get("http-equiv")
            content = tag.get("content")
            if http_equiv and content:
                meta_data[f"http-equiv-{http_equiv}"] = content
        
        return meta_data
    
    def _extract_title_tag(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the HTML title tag content."""
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else None
    
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all heading tags (h1-h6)."""
        headings = {}
        
        for i in range(1, 7):
            tag_name = f"h{i}"
            heading_tags = soup.find_all(tag_name)
            if heading_tags:
                headings[tag_name] = [h.get_text(strip=True) for h in heading_tags if h.get_text(strip=True)]
        
        return headings
    
    def _extract_open_graph(self, soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
        """Extract Open Graph meta tags."""
        og_data = {}
        
        # Find all meta tags and filter for og: properties
        all_meta_tags = soup.find_all("meta")
        og_tags = [tag for tag in all_meta_tags 
                  if tag.get("property") and tag.get("property").startswith("og:")]
        
        for tag in og_tags:
            property_name = tag.get("property")
            content = tag.get("content")
            
            if property_name and content:
                # Remove 'og:' prefix
                key = property_name.replace("og:", "")
                
                # Resolve URLs for image, video, audio
                if key in ["image", "video", "audio", "url"]:
                    content = self._resolve_url(content, base_url) or content
                
                og_data[key] = content
        
        return og_data
    
    def _extract_twitter_cards(self, soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
        """Extract Twitter Card meta tags."""
        twitter_data = {}
        
        # Find all meta tags and filter for twitter: attributes
        all_meta_tags = soup.find_all("meta")
        twitter_tags = [tag for tag in all_meta_tags 
                       if tag.get("name") and tag.get("name").startswith("twitter:")]
        
        for tag in twitter_tags:
            name = tag.get("name")
            content = tag.get("content")
            
            if name and content:
                # Remove 'twitter:' prefix
                key = name.replace("twitter:", "")
                
                # Resolve URLs for image
                if key == "image":
                    content = self._resolve_url(content, base_url) or content
                
                twitter_data[key] = content
        
        return twitter_data
    
    def _extract_canonical(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Extract canonical URL."""
        canonical_tag = soup.find("link", rel="canonical")
        if canonical_tag:
            href = canonical_tag.get("href")
            return self._resolve_url(href, base_url) if href else None
        return None
    
    def _extract_robots_meta(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract robots meta tag directives."""
        robots_data = {}
        
        # Standard robots meta tag
        robots_tag = soup.find("meta", name="robots")
        if robots_tag:
            content = robots_tag.get("content")
            if content:
                robots_data["robots"] = content
        
        # Googlebot specific
        googlebot_tag = soup.find("meta", name="googlebot")
        if googlebot_tag:
            content = googlebot_tag.get("content")
            if content:
                robots_data["googlebot"] = content
        
        return robots_data
    
    def _extract_json_ld(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract JSON-LD structured data."""
        json_ld_data = []
        
        # Find all script tags with type application/ld+json
        json_ld_scripts = soup.find_all("script", type="application/ld+json")
        
        for script in json_ld_scripts:
            try:
                json_content = json.loads(script.string)
                json_ld_data.append(json_content)
            except (json.JSONDecodeError, TypeError):
                # Skip invalid JSON
                continue
        
        return json_ld_data
    
    def _extract_hreflang(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract hreflang alternate language links."""
        hreflang_data = []
        
        hreflang_links = soup.find_all("link", rel="alternate", hreflang=True)
        
        for link in hreflang_links:
            hreflang = link.get("hreflang")
            href = link.get("href")
            
            if hreflang and href:
                resolved_href = self._resolve_url(href, base_url)
                if resolved_href:
                    hreflang_data.append({
                        "hreflang": hreflang,
                        "href": resolved_href
                    })
        
        return hreflang_data
    
    def _extract_icons(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        """Extract favicon, Apple touch icons, and manifest files."""
        icons_data = {}
        
        # Favicon
        favicon_link = soup.find("link", rel=["icon", "shortcut icon"])
        if favicon_link:
            href = favicon_link.get("href")
            if href:
                icons_data["favicon"] = self._resolve_url(href, base_url) or href
        
        # Apple touch icons
        apple_icons = soup.find_all("link", rel=lambda x: x and "apple-touch-icon" in x.lower())
        if apple_icons:
            apple_touch_icons = []
            for icon in apple_icons:
                href = icon.get("href")
                sizes = icon.get("sizes")
                if href:
                    icon_data = {"href": self._resolve_url(href, base_url) or href}
                    if sizes:
                        icon_data["sizes"] = sizes
                    apple_touch_icons.append(icon_data)
            icons_data["apple_touch_icons"] = apple_touch_icons
        
        # Manifest
        manifest_link = soup.find("link", rel="manifest")
        if manifest_link:
            href = manifest_link.get("href")
            if href:
                icons_data["manifest"] = self._resolve_url(href, base_url) or href
        
        return icons_data
    
    def _extract_dns_prefetch(self, soup: BeautifulSoup) -> List[str]:
        """Extract DNS prefetch hints."""
        dns_prefetch_links = soup.find_all("link", rel="dns-prefetch")
        return [link.get("href") for link in dns_prefetch_links if link.get("href")]
    
    def _resolve_url(self, url: Optional[str], base_url: str) -> Optional[str]:
        """
        Resolve relative URLs to absolute URLs.
        
        Args:
            url: The URL to resolve (may be relative)
            base_url: The base URL to resolve against
            
        Returns:
            Absolute URL or None if invalid
        """
        if not url:
            return None
        
        try:
            return urljoin(base_url, url)
        except Exception:
            return None
    
    async def scrape_to_json(self, url: str, indent: int = 2) -> str:
        """
        Scrape URL and return data as JSON string.
        
        Args:
            url: URL to scrape
            indent: JSON indentation
            
        Returns:
            JSON string of scraped data
        """
        data = await self.scrape_url(url)
        return json.dumps(data, ensure_ascii=False, indent=indent)
    
    async def close(self):
        """Close the async client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example of how to use the scraper
        scraper = DefaultWebScraper()
        
        try:
            # Replace with actual URL
            url = "https://example.com/article"
            data = await scraper.scrape_url(url)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await scraper.close()
        
        # Or use as async context manager
        # async with DefaultWebScraper() as scraper:
        #     data = await scraper.scrape_url("https://example.com/article")
        #     print(await scraper.scrape_to_json("https://example.com/article"))
    
    asyncio.run(main())

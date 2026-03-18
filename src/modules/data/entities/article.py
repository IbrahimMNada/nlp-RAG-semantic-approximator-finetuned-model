from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing import List, Dict, Any, Optional
import json
from ....core.base import Base


class Article(Base):
    """
    Anemic SQLAlchemy entity representing an article with content and SEO information
    Contains only data structure without business logic
    """
    __tablename__ = 'articles'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Article content
    title = Column(String(500), nullable=False, index=True)
    url = Column(String(2000), nullable=False, unique=True, index=True)
    author = Column(String(255), nullable=True, index=True)
    last_update = Column(String(255), nullable=True)
    
    # Flattened SEO meta fields
    seo_meta_description = Column(Text, nullable=True)
    seo_meta_keywords = Column(String(1000), nullable=True, index=True)
    seo_meta_viewport = Column(String(100), nullable=True)
    seo_meta_charset = Column(String(50), nullable=True)
    seo_meta_theme_color = Column(String(20), nullable=True)
    seo_meta_application_name = Column(String(100), nullable=True)
    seo_meta_thumbnail = Column(String(500), nullable=True)
    seo_title_tag = Column(String(500), nullable=True)
    seo_canonical = Column(String(1000), nullable=True)
    
    # SEO structured data (stored as JSON)
    seo_headings = Column(JSON, nullable=True)  # {h1: [], h2: [], h3: [], h4: []}
    seo_open_graph = Column(JSON, nullable=True)  # {title, type, url, image, site_name, description}
    seo_twitter_cards = Column(JSON, nullable=True)  # {card, site, title, description, image}
    
    # Additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Computed fields for better querying
    word_count = Column(Integer, nullable=True, index=True)
    paragraph_count = Column(Integer, nullable=True, index=True)
    
    # Relationship to paragraphs
    paragraphs = relationship("ArticleParagraph", back_populates="article", 
                             order_by="ArticleParagraph.order_index", 
                             cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:50] if self.title else ''}...', paragraphs={self.paragraph_count})>"
    
    def __str__(self) -> str:
        return f"Article: {self.title} ({self.word_count} words, {self.paragraph_count} paragraphs)"

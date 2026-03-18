from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ....core.base import Base

class ArticleParagraph(Base):
    """
    Anemic SQLAlchemy entity representing individual article paragraphs
    """
    __tablename__ = 'article_paragraphs'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to Article
    article_id = Column(Integer, ForeignKey('articles.id'), nullable=False, index=True)
    
    # Paragraph content and order
    content = Column(Text, nullable=False)
    order_index = Column(Integer, nullable=False, index=True)  # To maintain paragraph order
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship back to Article
    article = relationship("Article", back_populates="paragraphs")
    
    def __repr__(self) -> str:
        return f"<ArticleParagraph(id={self.id}, article_id={self.article_id}, order={self.order_index})>"

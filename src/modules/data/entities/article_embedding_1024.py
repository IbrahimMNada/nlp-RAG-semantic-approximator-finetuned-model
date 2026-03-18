from sqlalchemy import ForeignKey, Column, Integer, Index
from pgvector.sqlalchemy import Vector
from ....core.base import Base


class ArticleEmbedding1024(Base):
    __tablename__ = 'article_embedding_1024'
    article_id = Column(Integer, ForeignKey('articles.id'), unique=True, primary_key=True, nullable=False, index=True)
    embedding = Column(Vector(1024))
    
    __table_args__ = (
        Index(
            'article_embedding_1024_idx',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )
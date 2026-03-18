from sqlalchemy import ForeignKey, Column, Integer, Index
from pgvector.sqlalchemy import Vector
from ....core.base import Base


class ParagraphEmbedding1024(Base):
    __tablename__ = 'paragraph_embeddings_1024'
    article_id = Column(Integer, ForeignKey('articles.id'), primary_key=True, nullable=False, index=True)
    paragraph_id = Column(Integer, ForeignKey('article_paragraphs.id'), primary_key=True, nullable =False, index=True)
    embedding = Column(Vector(1024))
    
    __table_args__ = (
        Index(
            'paragraph_embeddings_1024_embedding_idx',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )
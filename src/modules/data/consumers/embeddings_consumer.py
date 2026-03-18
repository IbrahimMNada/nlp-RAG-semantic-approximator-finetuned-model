"""
RabbitMQ consumer for processing embeddings generation tasks.
This consumer listens to the embeddings queue and generates embeddings for articles.
"""
import asyncio
import json
import logging
import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from ....core.config import get_settings
from ....core.database import get_async_db_session
from ..services.embedding_service import get_embedding_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_embeddings(message_body: dict) -> None:
    """
    Process embeddings generation for a single article using EmbeddingService.
    
    Args:
        message_body: Dictionary containing article_id and paragraphs data
    """
    article_id = message_body.get("article_id")
    paragraphs = message_body.get("paragraphs", [])
    
    logger.info(f"Processing embeddings for article {article_id} with {len(paragraphs)} paragraphs")
    
    try:
        # Convert paragraphs format to match generate_and_save_all signature
        paragraph_data = [
            (p.get("paragraph_id"), p.get("content"))
            for p in paragraphs
        ]
        
        # Use EmbeddingService directly - no scraper needed for embedding-only operations
        embedding_service = get_embedding_service()
        
        async with get_async_db_session() as session:
            await embedding_service.generate_and_save_all(session, article_id, paragraph_data)
            
        logger.info(f"Successfully processed all embeddings for article {article_id}")
            
    except Exception as e:
        logger.error(f"Failed to process embeddings for article {article_id}: {str(e)}", exc_info=True)
        raise


async def on_message(message: AbstractIncomingMessage) -> None:
    """
    Callback for processing incoming RabbitMQ messages.
    
    Args:
        message: Incoming message from RabbitMQ
    """
    async with message.process():
        try:
            # Decode and parse message
            message_body = json.loads(message.body.decode())
            logger.info(f"Received message: {message_body}")
            
            # Process embeddings
            await process_embeddings(message_body)
            
            logger.info("Message processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            # Message will be requeued if processing fails
            raise


async def main() -> None:
    """
    Main consumer loop - connects to RabbitMQ and starts consuming messages.
    """
    settings = get_settings()
    
    logger.info("Starting embeddings consumer...")
    logger.info(f"Connecting to RabbitMQ at {settings.RABBITMQ_URL}")
    logger.info(f"Listening to queue: {settings.RABBITMQ_QUEUE_NAME}")
    
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
    
    async with connection:
        # Create channel
        channel = await connection.channel()
        
        # Set prefetch count to process one message at a time
        await channel.set_qos(prefetch_count=1)
        
        # Declare queue (idempotent)
        queue = await channel.declare_queue(
            settings.RABBITMQ_QUEUE_NAME,
            durable=True
        )
        
        logger.info("Consumer ready. Waiting for messages...")
        
        # Start consuming messages
        await queue.consume(on_message)
        
        # Keep the consumer running
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Consumer shutdown complete")

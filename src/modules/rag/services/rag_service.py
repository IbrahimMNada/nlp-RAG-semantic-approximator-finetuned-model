"""RAG service for chat and context augmentation."""
from typing import Optional
from ....core.base_dtos import ResponseDto
from ....contracts.data import SearchSimilarParagraphsRequestDto, SearchSimilarParagraphsResponseDto
from ..dtos.responses import AnswerResponseDto
from ....abstractions.interfaces.llm_provider_interface import ILLMProvider
from ....shared.event_bus import send as bus_send
from ....core.config import get_settings


class RagService:
    """
    Service for RAG (Retrieval-Augmented Generation) operations.
    Handles chat interactions with LLM providers and optional context augmentation
    from the data module.
    """
    
    def __init__(self, llm_provider: ILLMProvider):
        """Initialize RAG service.
        
        Args:
            llm_provider: LLM provider implementation (ChatGPT, Claude, etc.)
        """
        self.llm_provider = llm_provider
        self.settings = get_settings()
    
    async def search_context(
        self, 
        request: SearchSimilarParagraphsRequestDto
    ) -> ResponseDto[SearchSimilarParagraphsResponseDto]:
        """
        Search for similar paragraphs from the data module.
        
        Args:
            request: Search request containing text query and parameters
            
        Returns:
            ResponseDto containing similar paragraphs with metadata
        """
        # Call data module via event bus
        response = await bus_send(
            "search_similar_paragraphs",
            request=request,
        )
        
        return response
    
    async def ask_with_context(
        self,
        question: str,
        limit: int = 3,
        similarity_threshold: float = 0.5,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> ResponseDto[AnswerResponseDto]:
        """
        Search for context and ask ChatGPT a question based on that context.
        
        This is the typical RAG pattern: the user's question is used both to search
        for relevant context and as the question to ask ChatGPT.
        
        Args:
            question: Question to ask ChatGPT (also used to search for context)
            limit: Maximum number of paragraphs to retrieve
            similarity_threshold: Minimum similarity score
            temperature: ChatGPT temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            ResponseDto containing ChatGPT's answer with context sources
        """
        # Step 1: Search for relevant context using the question
        search_request = SearchSimilarParagraphsRequestDto(
            text=question,
            limit=limit,
            threshold=similarity_threshold,
            min_words=20
        )
        
        context_response = await self.search_context(search_request)
        
        sources = []
        context_text = ""
        
        # Step 2: Extract context if available
        if context_response.status_code == 200 and context_response.data:
            paragraphs = context_response.data.similar_paragraphs
            
            if paragraphs:
                context_parts = []
                for paragraph in paragraphs:
                    context_parts.append(
                        f"[From: {paragraph.article_title}]\n{paragraph.content}"
                    )
                    sources.append(paragraph.article_title)
                
                context_text = "\n\n".join(context_parts)
                sources = list(set(sources))  # Unique sources
        
        # Step 3: Build provider-specific prompt
        messages = self.llm_provider.build_prompt(
            context=context_text,
            question=question,
            has_context=bool(context_text),
        )
        
        # Step 4: Get response from LLM
        try:
            llm_response = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            assistant_message = llm_response["choices"][0]["message"]["content"]
            usage = llm_response.get("usage", {})
            tokens_used = usage.get("total_tokens")
            
            # Calculate cost
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost = self.llm_provider.calculate_cost(input_tokens, output_tokens)
            
            return ResponseDto[AnswerResponseDto].success(
                data=AnswerResponseDto(
                    message=assistant_message,
                    tokens_used=tokens_used,
                    context_used=bool(context_text),
                    sources=sources,
                    cost=cost,
                )
            )
            
        except Exception as e:
            return ResponseDto[AnswerResponseDto].fail(
                status_code=500,
                error_description=f"Error calling LLM provider: {str(e)}"
            )
    
    async def _fetch_context(self, query: str) -> Optional[dict]:
        """
        Fetch relevant context from the data module.
        
        Args:
            query: Search query for finding relevant context
            
        Returns:
            Dict containing context and sources, or None if unavailable
        """
        try:
            # Call the data module via event bus
            search_request = SearchSimilarParagraphsRequestDto(
                text=query,
                limit=3,
                threshold=0.0,
                min_words=10,
            )
            response = await bus_send(
                "search_similar_paragraphs",
                request=search_request,
            )
            
            # Check if we have valid data
            if response.status_code == 0 and response.data:
                paragraphs = response.data.similar_paragraphs
                
                # Build context string from paragraphs
                context_parts = []
                sources = []
                
                for paragraph in paragraphs:
                    context_parts.append(paragraph.content)
                    sources.append(paragraph.article_title)
                
                return {
                    "context": "\n\n".join(context_parts),
                    "sources": list(set(sources)),  # Unique sources
                }
            
            return None
            
        except Exception as e:
            # Log the error but continue without context
            print(f"Error fetching context: {e}")
            return None

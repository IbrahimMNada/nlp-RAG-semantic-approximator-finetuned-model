"""Service for SEO content generation using transformers model."""
import logging
from typing import Optional
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch

from ....core.config import get_settings
from ....shared.arabic_text_processor import ArabicTextProcessor

logger = logging.getLogger(__name__)


class SeoService:
    """Service for generating SEO content using the specified transformer model."""
    
    MODEL_NAME = "ibrahim-nada/cmdr7b-ar-seo-qlora-v1-2025-12-20_19.08.11"
    MODEL_REVISION = "3e1ed453ac9b13a87c5b4ac507a15dde646a8dea" 
    #MODEL_REVISION = "e6385f2707ab2bfdae078c553aa4baffa393eccb"  
    BASE_MODEL_NAME = "CohereLabs/c4ai-command-r7b-arabic-02-2025"
    MAX_TOKENS = 1020  # Maximum context length (leaving room for EOS and special tokens)
    
    def __init__(self):
        """Initialize SEO service with model and tokenizer."""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialized = False
        self.settings = get_settings()
        self.arabic_processor = ArabicTextProcessor()
        self._hf_logged_in = False
        
    def _initialize_model(self):
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return
            
        try:
            # Login to Hugging Face if not already logged in
            if not self._hf_logged_in and self.settings.HF_TOKEN:
                logger.info("Logging in to Hugging Face Hub")
                login(token=self.settings.HF_TOKEN)
                self._hf_logged_in = True
                logger.info("Successfully logged in to Hugging Face Hub")
            
            logger.info(f"Loading SEO model: {self.MODEL_NAME}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.BASE_MODEL_NAME,
                token=self.settings.HF_TOKEN if self.settings.HF_TOKEN else None,
                trust_remote_code=True,
             #   force_download=True,
                
            )
            
            # Set padding to right side
            self.tokenizer.padding_side = 'right'
            
            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate device and 4-bit quantization
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL_NAME,  # or whatever your base is
                quantization_config=quant_config,
                device_map="auto",
                token=self.settings.HF_TOKEN,
                local_files_only=True,
            )

            self.model = PeftModel.from_pretrained(
                base_model,
                self.MODEL_NAME,
                revision=self.MODEL_REVISION,              
                device_map="auto",
            )

            self.model.config.use_cache = False
            self.model.eval()
            self._initialized = True
            logger.info("SEO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SEO model: {str(e)}")
            raise RuntimeError(f"Failed to load SEO model: {str(e)}")
    
    async def generate_seo_content(
        self,
        text: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate SEO content from input text.
        
        Args:
            text: Input text for SEO content generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated SEO content
            
        Raises:
            RuntimeError: If model initialization or generation fails
        """
        try:
            # Initialize model on first use
            self._initialize_model()
            
            # Format prompt with SEO template
            formatted_prompt = (
                "<SEO_PROMPT_PREFIX>\n"
                "اكتب وصف ميتا للمقال التالي\n\n"
                "<ARTICLE_TEXT>\n"
                f"{text}\n\n"
                "<SEO_OUTPUT_PREFIX>\n"
            )
            
            logger.info(f"Original text length: {len(text)}")
            
            # Preprocess Arabic text
            preprocessed_text = self.arabic_processor.preprocess_arabic_text(formatted_prompt)
            logger.info(f"Preprocessed text length: {len(preprocessed_text)}")
            
            # Tokenize and truncate to MAX_TOKENS (accounting for special tokens)
            tokens = self.tokenizer.encode(preprocessed_text, add_special_tokens=True)
            original_token_count = len(tokens)
            
            # Account for special tokens - MAX_TOKENS includes them
            if original_token_count > self.MAX_TOKENS:
                logger.warning(f"Input exceeds {self.MAX_TOKENS} tokens ({original_token_count}). Truncating...")
                # Truncate keeping the special tokens at the start
                tokens = tokens[:self.MAX_TOKENS]
                preprocessed_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                logger.info(f"Truncated to {len(tokens)} tokens (including special tokens)")
            else:
                logger.info(f"Input token count: {original_token_count} (within limit)")
                preprocessed_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Generate text using model
            inputs = self.tokenizer(preprocessed_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=25)
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, prompt_len:]

            generated_text =  self.tokenizer.decode(generated_ids)
            
            logger.info(f"Generated SEO content length: {len(generated_text)}")
            
            return generated_text.replace("<|END_OF_TURN_TOKEN|>", "").strip()
            
        except Exception as e:
           # logger.error(f"Error generating SEO content: {str(e)}")
            raise RuntimeError(f"Failed to generate SEO content: {str(e)}")
    
    def get_model_name(self) -> str:
        """Get the model name being used."""
        return self.MODEL_NAME

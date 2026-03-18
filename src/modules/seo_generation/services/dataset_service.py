"""Service for loading and sampling from the xyz SEO dataset."""
import logging
import random
from typing import List, Dict, Any
from datasets import load_dataset
from huggingface_hub import login

from ....core.config import get_settings

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for loading and sampling from HF datasets."""
    
    DATASET_NAME = "ibrahim-nada/xyz-seo-data"
    
    def __init__(self):
        """Initialize dataset service."""
        self.settings = get_settings()
        self.dataset = None
        self._hf_logged_in = False
        self._initialized = False
    
    def _initialize_dataset(self):
        """Lazy initialization of dataset."""
        if self._initialized:
            return
        
        try:
            # Login to Hugging Face if not already logged in
            if not self._hf_logged_in and self.settings.HF_TOKEN:
                logger.info("Logging in to Hugging Face Hub")
                login(token=self.settings.HF_TOKEN)
                self._hf_logged_in = True
                logger.info("Successfully logged in to Hugging Face Hub")
            
            logger.info(f"Loading dataset: {self.DATASET_NAME}")
            
            # Load dataset from Hugging Face
            self.dataset = load_dataset(
                self.DATASET_NAME,
                token=self.settings.HF_TOKEN if self.settings.HF_TOKEN else None,
                split="validation"  # Adjust split name if needed
            )
            
            self._initialized = True
            logger.info(f"Dataset loaded successfully. Total samples: {len(self.dataset)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise RuntimeError(f"Failed to load dataset: {str(e)}")
    
    async def get_random_samples(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Get random samples from the dataset with 450 words in prompt column.
        
        Args:
            num_samples: Number of random samples to retrieve (default: 10)
            
        Returns:
            List of random dataset samples filtered by word count
            
        Raises:
            RuntimeError: If dataset initialization fails
            ValueError: If num_samples exceeds dataset size
        """
        try:
            # Initialize dataset on first use
            self._initialize_dataset()
            
            # Filter dataset to only items with 450 words in prompt column
            filtered_indices = []
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                if 'prompt' in sample:
                    word_count = len(sample['prompt'].split())
                    if word_count <= 450:
                        filtered_indices.append(idx)
            
            logger.info(f"Found {len(filtered_indices)} samples with 450 words in prompt")
            
            if len(filtered_indices) == 0:
                logger.warning("No samples found with exactly 450 words in prompt")
                return []
            
            # Adjust num_samples if it exceeds filtered dataset size
            actual_num_samples = min(num_samples, len(filtered_indices))
            
            if num_samples > len(filtered_indices):
                logger.warning(
                    f"Requested {num_samples} samples but only {len(filtered_indices)} have 450 words. "
                    f"Returning {actual_num_samples} samples."
                )
            
            # Get random indices from filtered list
            random_indices = random.sample(filtered_indices, actual_num_samples)
            
            # Get samples at those indices and filter out 'id' field
            samples = []
            for idx in random_indices:
                sample = dict(self.dataset[idx])
                # Remove 'id' field if it exists
                sample.pop('id', None)
                samples.append(sample)
            
            logger.info(f"Retrieved {len(samples)} random samples with 450 words")
            
            return samples
            
        except Exception as e:
            logger.error(f"Error getting random samples: {str(e)}")
            raise RuntimeError(f"Failed to get random samples: {str(e)}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        self._initialize_dataset()
        
        return {
            "name": self.DATASET_NAME,
            "size": len(self.dataset),
            "features": list(self.dataset.features.keys()) if self.dataset else [],
            "split": "train"
        }

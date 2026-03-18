"""Response DTO for dataset samples."""
from typing import List, Dict, Any
from pydantic import BaseModel


class DatasetSamplesResponseDto(BaseModel):
    """Response model for dataset samples."""
    
    samples: List[Dict[str, Any]]
    count: int
    dataset_name: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "samples": [
                    {"text": "Example text", "label": "Example label"},
                    {"text": "Another example", "label": "Another label"}
                ],
                "count": 2,
                "dataset_name": "ibrahim-nada/xyz-seo-data"
            }
        }
    }

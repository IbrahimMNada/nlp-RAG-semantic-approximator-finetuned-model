"""SEO generation module routes."""
from fastapi import APIRouter, Query, Request, Depends
from ...core.base_dtos import ResponseDto
from .dtos.requests.generate_seo_request_dto import GenerateSeoRequestDto
from .dtos.responses.generate_seo_response_dto import GenerateSeoResponseDto
from .dtos.responses.dataset_samples_response_dto import DatasetSamplesResponseDto
from .dependencies import SeoServiceDep, DatasetServiceDep
from ...core.security import limiter, verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/generate", response_model=ResponseDto[GenerateSeoResponseDto])
@limiter.limit("10/minute")
async def generate_seo_content(
    request: Request,
    generate_request: GenerateSeoRequestDto,
    seo_service: SeoServiceDep,
):
    """
    Generate SEO content using transformer model.
    
    This endpoint uses the cmdr7b-ar-seo-qlora-v1-2025-12-20_19.08.11 model
    to generate SEO-optimized content from the input text.
    
    Args:
        generate_request: Request containing text and generation parameters
        seo_service: Injected SEO service
        
    Returns:
        ResponseDto containing generated SEO content
    """
    # Generate SEO content
    generated_text = await seo_service.generate_seo_content(
        text=generate_request.text,
        max_length=generate_request.max_length,
        temperature=generate_request.temperature,
        top_p=generate_request.top_p,
    )
    
    # Create response DTO
    response_data = GenerateSeoResponseDto(
        generated_text=generated_text,
        input_text=generate_request.text,
        model_name=seo_service.get_model_name()
    )
    
    return ResponseDto(
        status_code=200,
        error_description=None,
        data=response_data
    )


@router.get("/dataset/random-samples", response_model=ResponseDto[DatasetSamplesResponseDto])
async def get_random_dataset_samples(
    num_samples: int = Query(default=10, ge=1, le=100, description="Number of random samples to retrieve"),
    dataset_service: DatasetServiceDep = None,
):
    """
    Get random samples from the xyz SEO dataset.
    
    This endpoint loads the ibrahim-nada/xyz-seo-data dataset from Hugging Face
    and returns a specified number of random samples.
    
    Args:
        num_samples: Number of random samples to retrieve (1-100, default: 10)
        dataset_service: Injected dataset service
        
    Returns:
        ResponseDto containing random dataset samples
    """
    # Get random samples
    samples = await dataset_service.get_random_samples(num_samples=num_samples)
    
    # Create response DTO
    response_data = DatasetSamplesResponseDto(
        samples=samples,
        count=len(samples),
        dataset_name=dataset_service.DATASET_NAME
    )
    
    return ResponseDto(
        status_code=200,
        error_description=None,
        data=response_data
    )

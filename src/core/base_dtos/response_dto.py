from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T")

class ResponseDto(BaseModel, Generic[T]):
    data: Optional[T] = None
    status_code: int = 0
    error_description: Optional[str] = None

    @classmethod
    def success(cls, data: T) -> "ResponseDto[T]":
        return cls(data=data, status_code=200, error_description=None)

    @classmethod
    def fail(cls, status_code: int, error_description: str) -> "ResponseDto[T]":
        return cls(data=None, status_code=status_code, error_description=error_description)
    
    class Config:
        from_attributes = True 
        extra = "ignore"

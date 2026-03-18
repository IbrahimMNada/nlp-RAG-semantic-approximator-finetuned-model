from pydantic import BaseModel, HttpUrl

class ProcessFileDto(BaseModel):
    url: HttpUrl
    class Config:
        from_attributes = True
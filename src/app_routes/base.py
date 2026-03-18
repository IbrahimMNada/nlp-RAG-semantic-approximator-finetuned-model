from fastapi import APIRouter , Depends
from ..core import get_settings , Settings
base_router = APIRouter()

@base_router.get("/ping")
async def read_root(settings  : Settings = Depends(get_settings)):
    return {"Bong From ": settings.APP_NAME}
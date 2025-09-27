from fastapi import APIRouter
from app.utils.countries import ISO_COUNTRIES

router = APIRouter()

@router.get("/countries", tags=["Meta"])
def list_countries():
    return ISO_COUNTRIES

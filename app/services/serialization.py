"""Serialization helpers to enrich ORM objects with computed fields."""
from typing import Any, Dict
from app.models.models import Image


def image_to_dict(image: Image) -> Dict[str, Any]:
    # Provide backward compatibility: "image_url" now returns absolute full_url so existing
    # frontend code that uses image.image_url gets a directly fetchable link. The original
    # relative path is exposed as "relative_path" for any future migration logic.
    return {
        "id": image.id,
        "case_id": image.case_id,
        "image_url": image.full_url,          # upgraded
        "relative_path": image.image_url,     # original stored value
        "full_url": image.full_url,
    }

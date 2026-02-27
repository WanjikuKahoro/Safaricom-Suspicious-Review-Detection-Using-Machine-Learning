from pydantic import BaseModel, Field
from typing import Optional

class ReviewFeatures(BaseModel):
    review_text: str
    rating: int = Field(..., ge=1, le=5)
    thumbs_up: Optional[int] = Field(0, ge=0)
    is_code_mixed: Optional[bool] = False
    is_sheng_like: Optional[bool] = False

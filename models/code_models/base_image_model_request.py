from pydantic import BaseModel

from settings import SELECTED_MODEL

class BaseImageModelRequest(BaseModel):
    model: str = SELECTED_MODEL

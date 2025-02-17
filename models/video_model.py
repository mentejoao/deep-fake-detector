from pydantic import BaseModel

class VideoModel(BaseModel):
    filename: str
    content_type: str
    format_type: str
    size: int
    duration: int
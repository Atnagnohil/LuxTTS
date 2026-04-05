from pydantic import BaseModel, Field
from typing import List


class VoiceInfo(BaseModel):
    voice_id: str
    label: str
    type: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')
    num_steps: int = Field(default=4, ge=1, le=20)
    t_shift: float = Field(default=0.9, ge=0.1, le=1.0)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    guidance_scale: float = Field(default=3.0, gt=0)


class VoicesResponse(BaseModel):
    voices: List[VoiceInfo]
    total: int


class CloneVoiceResponse(BaseModel):
    voice_id: str
    message: str

from pydantic import BaseModel, Field
from fastapi import UploadFile, File
from typing import Optional, List

class Image2ImageRequest(BaseModel):
    prompt: str
    height: int = 600
    width: int = 600
    strength: float = .1
    guidance: float = .9
    pipeline: str = ""
    lora_choice: str = "None"
    negative_prompt: str = "easynegative, human, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,"

import io

from fastapi import APIRouter, Response, UploadFile, Depends, File

from settings import MODEL_DIRECTORY

from models.code_models.image_to_image_request import Image2ImageRequest
from models.code_models.abstract_image_to_image_pipeline import AbstractImageToImagePipeline

from routers.v2 import find_lora_by_name, __clean_up_pipeline

ROUTER = APIRouter(
    prefix="/img2img",
    tags=["Image To Image Generation"]
)

@ROUTER.post("/generate/")
def generate_picture_from_picture(image: Image2ImageRequest = Depends(), uploaded_image: UploadFile=File(...)):
    # TODO: Dreambooth instead of base_lora
    contextual_lora = find_lora_by_name(
        image.lora_choice,
        image.pipeline
        )
    pipeline = AbstractImageToImagePipeline(
        MODEL_DIRECTORY, 
        image.pipeline)
    image_store = io.BytesIO()
    for generated_image in pipeline.generate_image(
        image.prompt,
        uploaded_image,
        image.strength,
        image.guidance,
        height=image.height,
        width=image.width,
        negative_prompt=image.negative_prompt,
        lora_choice=contextual_lora):
        generated_image.save(image_store,"png")
        break

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)

    return Response(content=image_store.getvalue(), media_type="image/png")

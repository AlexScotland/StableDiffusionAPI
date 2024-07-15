import io
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline,StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusionXLControlNetPipeline, StableVideoDiffusionPipeline
from fastapi import APIRouter, Response, UploadFile, Depends, File
from fastapi.responses import FileResponse

from image_ai import ImageGeneratorStability
from video_ai import VideoGeneratorStability
from settings import MODEL_DIRECTORY, SELECTED_MODEL, SELECTED_VIDEO_MODEL

from models.code_models.image_request import ImageRequest
from models.code_models.image_to_image_request import Image2ImageRequest

MAIN_IMAGE_AI = ImageGeneratorStability(
    MODEL_DIRECTORY,
    SELECTED_MODEL
    )

api_router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)

@api_router.get("/loras/")
def all_lora_full_details():
    all_lora = []
    for lora in MAIN_IMAGE_AI.lora_list:
        all_lora.append(lora)
    return all_lora

@api_router.get("/loras/keywords")
def all_loras():
    all_lora_keywords = []
    for lora in MAIN_IMAGE_AI.lora_list:
        for keyword in lora.keywords:
            all_lora_keywords.append(keyword)
    return all_lora_keywords

@api_router.put("/download/")
def download_model_from_hugging_face(model_name: str):
    model = model_name
    pipeline = DiffusionPipeline.from_pretrained(model)
    pipeline.save_pretrained(f"{MODEL_DIRECTORY}/{model.split('/')[-1]}")
    return {"Status": "Downloaded"}

# @api_router.post("{model_id}/generate/")
# def generate_from_reference_picture():
#     # https://civitai.com/models/388607/realityfuse-xl
#     ai_generation = DiffusionPipeline.from_pretrained(f"{FULL_DIR}/models/stable-video-diffusion-img2vid")    
#     return {"Hello": "World"}

@api_router.post("/generate/")
def generate_picture(image: ImageRequest):
    image_store = io.BytesIO()
    for generated_image in MAIN_IMAGE_AI.generate_image(
        image.prompt,
        height=image.height,
        width=image.width,
        negative_prompt=image.negative_prompt,
        lora_choice=image.lora_choice):
        generated_image.save(image_store,"png")
        break

    return Response(content=image_store.getvalue(), media_type="image/png")


@api_router.post("/generate_video/")
def generate_video(file: UploadFile):
    frames = MAIN_VIDEO_AI.generate_video(file)
    return FileResponse(frames, media_type='video/mp4',filename="generated.mp4")

@api_router.post("/generate_from_image/")
def generate_picture_from_image(prompt: Image2ImageRequest = Depends(), image: UploadFile=File(...)):
    image_store = io.BytesIO()
    request_object_content = image.file.read()
    img = Image.open(io.BytesIO(request_object_content))
    for generated_image in IMG2IMAGE.generate_image(
        prompt.prompt,
        img,
        height=prompt.height,
        width=prompt.width,
        negative_prompt=prompt.negative_prompt,
        lora_choice=prompt.lora_choice):
        generated_image.save(image_store,"png")
        break

    return Response(content=image_store.getvalue(), media_type="image/png")

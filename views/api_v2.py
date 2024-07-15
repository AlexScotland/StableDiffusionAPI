import io
import gc
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline,StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusionXLControlNetPipeline, StableVideoDiffusionPipeline
from fastapi import Response, UploadFile, Depends, File
from fastapi.responses import FileResponse
from fastapi import APIRouter

from factories.lora_factory import LoRAFactory
from models.LoRA.lora_conf import ALL_LORAS
from models.code_models.base_image_pipeline import BaseImagePipeline
from settings import MODEL_DIRECTORY, SELECTED_MODEL, SELECTED_VIDEO_MODEL, BASE_DIR

from models.code_models.base_image_request import BaseImageRequest

V2_API_ROUTER = APIRouter(
    prefix="/v2/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)

#  TODO: Add better way to dynamically change this PIPELINE

def create_all_loras():
    all_loras = []
    for lora_path in ALL_LORAS:
        all_loras.append(LoRAFactory.create(ALL_LORAS[lora_path], lora_path))
    return all_loras
    
LORAS = create_all_loras()

def find_lora_by_name(name):

    for lora in LORAS:
        if lora == name and lora.base_model in SELECTED_MODEL:
            return lora
    return None

@V2_API_ROUTER.get("/loras/")
def all_lora_full_details():
    ret_list = []
    for lora in LORAS:
        if lora.base_model in SELECTED_MODEL:
            ret_list.append(lora)
    return ret_list

@V2_API_ROUTER.post("/generate/")
def generate_picture(image: BaseImageRequest):
    # TODO: Dreambooth instead of base_lora
    base_lora = find_lora_by_name(image.base_lora)
    contextual_lora = find_lora_by_name(image.contextual_lora)
    pipeline = BaseImagePipeline(MODEL_DIRECTORY, SELECTED_MODEL, base_lora=base_lora)
    image_store = io.BytesIO()
    for generated_image in pipeline.generate_image(
        image.prompt,
        height=image.height,
        width=image.width,
        negative_prompt=image.negative_prompt,
        lora_choice=contextual_lora):
        generated_image.save(image_store,"png")
        break
    # Cleanup the pipeline
    del pipeline
    # Run garbage collection
    gc.collect()

    # Clear GPU memory cache
    torch.cuda.empty_cache()
    return Response(content=image_store.getvalue(), media_type="image/png")

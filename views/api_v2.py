import io
import gc
import torch
import os

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from fastapi import Response, UploadFile, Depends, File
from fastapi.responses import FileResponse
from fastapi import APIRouter

from factories.lora_factory import LoRAFactory
from models.LoRA.lora_conf import ALL_LORAS
from models.code_models.base_image_pipeline import BaseImagePipeline
from models.code_models.abstract_image_pipeline import AbstractImagePipeline
from settings import MODEL_DIRECTORY, SELECTED_MODEL, SELECTED_VIDEO_MODEL, BASE_DIR

from models.code_models.base_image_request import BaseImageRequest
from models.code_models.base_image_model_request import BaseImageModelRequest

V2_API_ROUTER = APIRouter(
    prefix="/v2/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)


def __clean_up_pipeline(pipeline):
    # Cleanup the pipeline
    del pipeline
    # Run garbage collection
    gc.collect()

    # Clear GPU memory cache
    torch.cuda.empty_cache()

def create_all_loras():
    all_loras = []
    for lora_path in ALL_LORAS:
        all_loras.append(LoRAFactory.create(ALL_LORAS[lora_path], lora_path))
    return all_loras
    
LORAS = create_all_loras()

def find_lora_by_name(name, model):
    for lora in LORAS:
        if lora == name and lora.base_model in model:
            return lora
    return None

@V2_API_ROUTER.post("/loras/")
def all_lora_full_details(image: BaseImageModelRequest):
    ret_list = []
    for lora in LORAS:
        if lora.base_model in image.model:
            ret_list.append(lora)
    return ret_list

@V2_API_ROUTER.get("/models/")
def get_all_models():
    model_dir = MODEL_DIRECTORY
    if not model_dir.endswith("/"): model_dir += "/"
    return [f for f in os.listdir(model_dir) if os.path.isdir(model_dir+f)]


@V2_API_ROUTER.put("/download/")
def download_model_from_hugging_face(model_name: str):
    pipeline = DiffusionPipeline.from_pretrained(model_name)
    pipeline.save_pretrained(f"{MODEL_DIRECTORY}/{model_name.split('/')[-1]}")

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)
    return {"Status": "Downloaded"}

@V2_API_ROUTER.post("/export/safetensor")
def export_safetensor_local(safetensor_name: str):
    # TODO: Move this to another export directory,
    # have url and filetype be same variable
    extension = ".safetensors"
    if not safetensor_name.endswith(extension):
        safetensor_name+=extension
    
    safetensor_directory = f"{MODEL_DIRECTORY}/{safetensor_name}"
    model_export_directory = safetensor_directory.replace(extension, "")
    try:
        pipeline = StableDiffusionXLPipeline.from_single_file(
        safetensor_directory,
        local_files_only=True,
        use_safetensors=True
        )
    except TypeError as type_error_message:
        if "tokenizer" or "encoder" in type_error_message:
            try:
                pipeline = StableDiffusionPipeline.from_single_file(
                    safetensor_directory,
                    local_files_only=True,
                    use_safetensors=True
                    )
            except TypeError as type_error_message:
                pipeline = FluxPipeline.from_single_file(
                    safetensor_directory,
                    local_files_only=True,
                    use_safetensors=True
                    )
        else:
            raise type_error_message
    pipeline.save_pretrained(model_export_directory)

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)

    # Remove the old safe tensor
    os.remove(safetensor_directory)
    return {"model": safetensor_name.replace(extension, "")}

@V2_API_ROUTER.post("/generate/")
def generate_picture(image: BaseImageRequest):
    # TODO: Dreambooth instead of base_lora
    base_lora = find_lora_by_name(
        image.base_lora,
        image.model
        )
    contextual_lora = find_lora_by_name(
        image.contextual_lora,
        image.model
        )
    pipeline = AbstractImagePipeline(
        MODEL_DIRECTORY,
        image.model,
        base_lora=base_lora,
        diffuser=AutoPipelineForText2Image)
    image_store = io.BytesIO()
    for generated_image in pipeline.generate_image(
        image.prompt,
        height=image.height,
        width=image.width,
        negative_prompt=image.negative_prompt,
        lora_choice=contextual_lora):
        generated_image.save(image_store,"png")
        break

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)

    return Response(content=image_store.getvalue(), media_type="image/png")
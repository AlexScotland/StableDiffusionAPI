import io

from fastapi import APIRouter, Response, UploadFile, Depends, File
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from diffusers import DDIMScheduler, AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterPlus

from settings import MODEL_DIRECTORY

from ..models.serializers.base_image import BaseImage
from ..models.image_pipeline import SDImagePipeline

from routers.v2 import __clean_up_pipeline


ROUTER = APIRouter(
    prefix="/base_ip_adapter",
    tags=["IP Adapters for Same character images"]
)

@ROUTER.post("/generate/")
def generate(image: BaseImage = Depends(), uploaded_image: UploadFile=File(...)):

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    image_pipeline = SDImagePipeline(
        MODEL_DIRECTORY,
        image.model,
        noise_scheduler_ddim=noise_scheduler,
        diffuser=StableDiffusionXLCustomPipeline
    )
    image_store = io.BytesIO()
    image = image_pipeline.create_image(
        prompt = image.prompt,
        image = Image.open(io.BytesIO(uploaded_image.file.read())),
        scale = 0.7,
        height=image.height,
        width=image.width)
    image[0].save(image_store,"png")

    # Cleanup our pipeline
    __clean_up_pipeline(image_pipeline)

    return Response(content=image_store.getvalue(), media_type="image/png")
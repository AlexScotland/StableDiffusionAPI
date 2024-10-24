from fastapi import APIRouter

from models.code_models.image_to_image_request import Image2ImageRequest

ROUTER = APIRouter(
    prefix="/img2img"
)


@ROUTER.post("/generate/")
def generate_picture_from_picture(image: Image2ImageRequest):
    # TODO: Dreambooth instead of base_lora
    base_lora = find_lora_by_name(
        image.base_lora,
        image.model
        )
    contextual_lora = find_lora_by_name(
        image.contextual_lora,
        image.model
        )
    pipeline = AbstractImagePipeline(MODEL_DIRECTORY, image.model, base_lora=base_lora)
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
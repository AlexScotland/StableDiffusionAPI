import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, DPMSolverMultistepScheduler

from image_ai import ImageGeneratorStability
from models.code_models.lora import LoRA
from models import ALL_LORAS

class BaseImagePipeline(ImageGeneratorStability):

    def __init__(
        self,
        model_dir,
        model_name,
        base_lora: LoRA=None,
        diffuser=StableDiffusionXLPipeline
        ):
            self.model_name = model_name
            if model_dir[-1] != '/':
                model_dir += '/'
            self.model_dir = model_dir
            self.pipeline = self._generate_pipeline(diffuser)
            self.base_lora = base_lora
            if self.base_lora:
                self.pipeline.load_lora_weights(
                            self.base_lora.path[0],
                            weight_name=self.base_lora.weight_name,
                            adapter_name="base"
                            )
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)

    def generate_image(
            self,
            prompt,
            height=600,
            width=600,
            lora_choice: LoRA=None,
            negative_prompt= "easynegative, human, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,",
            steps=50,
            number_of_images=1,
        ):
            if lora_choice:
                self.pipeline.load_lora_weights(
                            lora_choice.path[0],
                            weight_name=lora_choice.weight_name,
                            adapter_name="contextual"
                            )
                if self.base_lora:
                    self.pipeline.set_adapters(["base", "contextual"], adapter_weights=[self.base_lora.scale, lora_choice.scale])
                    self.pipeline.fuse_lora(adapter_names=["base", "contextual"], lora_scale=1.0)
                else:
                    self.pipeline.fuse_lora(adapter_names=["contextual"], lora_scale=1.0)
            elif self.base_lora:
                self.pipeline.fuse_lora(adapter_names=["base"], lora_scale=1.0)
            self.pipeline.unload_lora_weights()
            generated_images = self.pipeline(
                                    prompt,
                                    height=height,
                                    width=width,
                                    negative_prompt=negative_prompt,
                                    num_images_per_prompt=number_of_images
                                    ).images
            # Teardown
            if lora_choice:
                self.pipeline.unfuse_lora()
                self.pipeline.unload_lora_weights()
            return generated_images
import io
import os
import torch
from fastapi import APIRouter, Response
from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from models import ALL_LORAS
from models.code_models.lora import LoRA

class ImageToImageGeneratorStability:
    def __init__(
            self,
            model_dir,
            model_name,
            diffuser=AutoPipelineForImage2Image,
        ):
        self.model_name=model_name
        if model_dir[-1] != '/':
            model_dir += '/'
        self.model_dir=model_dir
        self.pipeline = self._generate_pipeline(diffuser)
        self.lora_list = self._generate_lora_objects(ALL_LORAS)
        # Apply scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)

    def _generate_pipeline(self, diffuser):
        # Default to CPU
        input_touch_type=torch.float32
        input_variant="fp32"
        to_value = "cpu"

        if torch.cuda.is_available():
            input_touch_type=torch.float16
            input_variant="fp16"
            to_value = "cuda"
        
        elif torch.backends.mps.is_available():
            input_touch_type=torch.float16
            input_variant="fp16"
            to_value = "mps"
            
        pipeline=diffuser.from_pretrained(
                self.model_dir+self.model_name,
                torch_dtype=input_touch_type, 
                variant=input_variant,
                use_safetensors=True,
                load_safety_checker=False,
                local_files_only=True
                ).to(to_value)
        return pipeline
    
    def _generate_lora_objects(self, path_to_all_loras):
        all_loras = []
        for lora_name in ALL_LORAS:
            all_loras.append(
                LoRA(
                    lora_name, 
                    ALL_LORAS[lora_name]['weight_name'],
                    ALL_LORAS[lora_name]['weight'],
                    ALL_LORAS[lora_name]['keywords']
                    )
                )
        return all_loras

    def generate_image(
        self,
        prompt,
        image,
        height=600,
        width=600,
        lora_choice="dynamic",
        negative_prompt= " ",
        steps=50,
        number_of_images=1,
        strength=.70,
        guidance_scale=30
    ):
        lora_to_use = None
        if isinstance(lora_choice, str):
            if lora_choice == "dynamic":
                lora_to_use = self.get_loras_by_keywords(prompt)
            else:
                lora_to_use = self.get_lora_by_name(lora_choice)
            if lora_to_use:
                for lora in lora_to_use:
                    lora_path = lora.path[0]
                    lora_weight_name = lora.weight_name
                    self.pipeline.load_lora_weights(
                        lora_path,
                        weight_name=lora_weight_name,
                        )
                    self.pipeline.fuse_lora()
            
        generated_images = self.pipeline(
            prompt,
            image=image,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=number_of_images,
            strength=strength,
            guidance_scale=guidance_scale
            ).images
        # Teardown
        if lora_to_use:
            self.pipeline.unfuse_lora()
            self.pipeline.unload_lora_weights()
        return generated_images
    
    def get_lora_by_name(self, lora_name):
        lora_list = []
        for lora in self.lora_list:
            if lora.weight_name.replace(".safetensors","").lower() == lora_name.lower():
                lora_list.append(lora)
                break
        return lora_list
    
    def get_loras_by_keywords(self, prompt):
        all_loras = set()
        for lora in self.lora_list:
            for keyword in lora.keywords:
                if keyword.lower() in prompt.lower():
                    all_loras.add(lora)
                    break
        return all_loras

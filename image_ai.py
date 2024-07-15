import io
import os
import torch
from fastapi import APIRouter, Response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from models import ALL_LORAS
from models.code_models.lora import LoRA

class ImageGeneratorStability:
    def __init__(
            self,
            model_dir,
            model_name,
            diffuser=DiffusionPipeline,
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
                    ALL_LORAS[lora_name]['keywords'],
                    scale=ALL_LORAS[lora_name]['scale']
                    )
                )
        return all_loras

    def generate_image(
        self,
        prompt,
        height=600,
        width=600,
        lora_choice="dynamic",
        negative_prompt= "easynegative, human, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,",
        steps=50,
        number_of_images=1,
    ):
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
        print(f"We are Generating for Prompt: {prompt}")
        print(f"We are using the following loras:")
        for lora in lora_to_use:
            print(lora.weight_name)
        generated_images = self.pipeline(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=number_of_images
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

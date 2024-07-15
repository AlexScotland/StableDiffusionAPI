import io
import os
import torch
from fastapi import APIRouter, Response
from diffusers import StableVideoDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image, export_to_video
from PIL import Image


class VideoGeneratorStability:
    def __init__(
            self,
            model_dir,
            model_name,
            diffuser=StableVideoDiffusionPipeline,
        ):
        self.model_name=model_name
        if model_dir[-1] != '/':
            model_dir += '/'
        self.model_dir=model_dir
        self.pipeline = self._generate_pipeline(diffuser)
        # # Apply scheduler
        # self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)

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
    

    def generate_video(
        self,
        image,
        num_inference_steps=25,
        motion_bucket_id=30,
        num_frames=25,
        height=320,
        width=512,
    ):
        image = Image.open(io.BytesIO(image.file.read()))
        image = load_image(image)
        image = image.resize ((1024, 576))
        frames = self.pipeline(
            image,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            motion_bucket_id=motion_bucket_id,
            decode_chunk_size=8,
            noise_aug_strength=0.1).frames[0]
        return export_to_video(frames)


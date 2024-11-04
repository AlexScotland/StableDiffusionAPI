import torch

from diffusers import StableDiffusionPipeline
from ip_adapter import IPAdapterPlusXL

from helpers.package import find_package_name

class SDImagePipeline():
    """
        Image pipeline used for IP Generation
    """

    def __init__(
                self,
                model_dir,
                model_name,
                vae_model=None,
                image_encoder_path=f"{find_package_name('ip_adapter').path}/image_models/ip_models/image_encoder/",
                ip_checkpoint=f"{find_package_name('ip_adapter').path}/image_models/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
                noise_scheduler_ddim=None,
                diffuser=StableDiffusionPipeline
                ):
        self._get_system_architecture()
        self.model_name = model_name
        if model_dir[-1] != '/':
            model_dir += '/'
        self.model_dir = model_dir
        self.noise_scheduler_ddim = noise_scheduler_ddim
        self.vae_model = vae_model
        self.pipeline = self._generate_pipeline(diffuser)
        
        self.ip_model = IPAdapterPlusXL(self.pipeline, image_encoder_path, ip_checkpoint, self.to_value, num_tokens=16)
    
    def _get_system_architecture(self):
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

        self.input_touch_type = input_touch_type
        self.input_variant = input_variant
        self.to_value = to_value


    def _generate_pipeline(self, diffuser):
            
        pipeline = diffuser.from_pretrained(
                self.model_dir+self.model_name,
                torch_dtype=self.input_touch_type, 
                variant=self.input_variant,
                use_safetensors=True,
                load_safety_checker=False,
                local_files_only=True
                )
        return pipeline

    def create_image(
                    self,
                    prompt,
                    scale,
                    width,
                    height,
                    image,
                    seed=420,
                    num_inference_steps=50):

        return self.ip_model.generate(
            pil_image = image,
            num_samples = 1,
            prompt = prompt,
            scale = scale,
            width = width,
            height = height,
            num_inference_steps = num_inference_steps,
            seed = seed
        )

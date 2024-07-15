import os

BASE_DIR = os.getenv("BASE_DIRECTORY", "Z:/Collaborative_AI_Chat/src/stable_diffusion_backend")
SELECTED_MODEL = os.getenv("SELECTED_MODEL","stable-video-diffusion")
SELECTED_VIDEO_MODEL = os.getenv("SELECTED_VIDEO_MODEL","stable-video-diffusion-img2vid-xt")

MODEL_DIRECTORY = f"{BASE_DIR}/models/full_models"
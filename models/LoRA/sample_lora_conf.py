# To add new LoRAs, we need to do a few things:
#  0. Create or rename to `lora_conf.py`
#  1. Download LoRA
#  2. Create a new Folder for this Lora (LORA_FOLDER)
#  3. Paste the Lora.safetensors into the folder (LORA_FILE_NAME.safetensors)
#  4. Add new JSON object for LoRA below
#  5. Add all keywords used in LoRA for trigger
#  6. Add the weight scale
#  7. Add if this is a style to be used as a base lora
#  8. Add compatible base_model


from settings import BASE_DIR

ALL_LORAS ={
    f'{BASE_DIR}/models/LoRA/LORA_FOLDER':{
        "weight_name":"LORA_FILE_NAME.safetensors",
        "keywords":["KEYWORD1","ANOTHER KEYWORD", "THIS KEYWORD TRIGGERS"],
        "scale": 1,
        "is_style": False,
        "model": "stable-diffusion-xl-base-1.0"
        },
    f'{BASE_DIR}/models/LoRA/LORA_FOLDER_2':{
        "weight_name":"ANOTHER_LORA.safetensors",
        "keywords":["KEYWORD1","ANOTHER KEYWORD", "THIS KEYWORD TRIGGERS"],
        "scale": .2,
        "is_style": True,
        "model": "stable-diffusion-xl-base-1.0"
        },
}

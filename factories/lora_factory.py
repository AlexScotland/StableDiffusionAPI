from models.code_models.lora import LoRA

class LoRAFactory:

    @staticmethod
    def create(json, path):
        return LoRA(
            path=path,
            weight_name=json.get("weight_name"),
            keywords=json.get("keywords"),
            base_model=json.get("model"),
            scale=json.get("scale"),
            is_style=json.get("is_style"),
        )

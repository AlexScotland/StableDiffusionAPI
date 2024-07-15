class LoRA():

    def __init__(
        self,
        path,
        weight_name,
        keywords,
        base_model,
        scale=1,
        is_style=False,
        ):
        self.path=path,
        self.weight_name=weight_name
        self.scale = scale
        self.keywords = set()
        self.is_style = is_style
        self.base_model = base_model
        for word in keywords:
            self.keywords.add(word.lower())
        
    def __eq__(self, weight_name):
        if isinstance(weight_name, str):
            return self.weight_name.replace(".safetensors","") == weight_name.replace(".safetensors","")
        return False
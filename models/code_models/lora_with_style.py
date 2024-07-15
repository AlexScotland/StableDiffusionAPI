class LoRA():

    def __init__(
        self,
        path,
        weight_name,
        keywords,
        style,
        scale = 1,
        ):
        self.path=path,
        self.weight_name=weight_name
        self.scale = scale
        self.style = style
        self.keywords = set()
        for word in keywords:
            self.keywords.add(word.lower())
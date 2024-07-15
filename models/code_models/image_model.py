class ImageModel():

    def __init__(
            self,
            model_name,
            model_directory
            ):
        self.model_name = model_name
        self.model_directory = model_directory
    
    def __eq__(self, model_name):
        if isinstance(model_name, str):
            return self.model_name == model_name
        return False

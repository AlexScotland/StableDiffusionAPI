from models.code_models.image_model import ImageModel

class ImageModelFactory:

    @staticmethod
    def create(model_name, model_directory):
        return ImageModel(
            model_name,
            model_directory
        )

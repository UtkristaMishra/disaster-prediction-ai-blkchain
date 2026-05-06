from .models.wildfire import WildfirePredictor
from .models.flood import FloodPredictor

class ModelRegistry:
    def __init__(self):
        self.models = {
            "wildfire": WildfirePredictor(),
            "flood": FloodPredictor(),
        }

    def get(self, model_name: str):
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")
        return model

    def predict(self, model_name: str, **features):
        model = self.get(model_name)
        # Forward the validated request fields unchanged so each model can use
        # the feature subset it was trained for without registry-level defaults.
        return model.predict(**features)

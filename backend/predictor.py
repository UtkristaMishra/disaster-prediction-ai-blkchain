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

    def predict(self, model_name: str, rainfall: float, temperature: float, humidity: float, wind_speed: float, ndvi: float, elevation: float):
        model = self.get(model_name)
        return model.predict(
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            ndvi=ndvi,
            elevation=elevation,
        )

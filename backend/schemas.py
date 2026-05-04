from pydantic import BaseModel

class PredictionRequest(BaseModel):
    rainfall: float
    temperature: float
    humidity: float
    wind_speed: float
    ndvi: float
    elevation: float

class PredictionResponse(BaseModel):
    risk_probability: float
    intensity: float
    risk_label: str
    confidence: float
    message: str

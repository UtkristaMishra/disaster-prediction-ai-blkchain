from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    rainfall: float = Field(..., ge=0, description="Recent rainfall in millimeters.")
    temperature: float = Field(..., ge=-60, le=70, description="Mean air temperature in degrees Celsius.")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity percentage.")
    wind_speed: float = Field(..., ge=0, description="Wind speed in kilometers per hour.")
    ndvi: float = Field(..., ge=-1, le=1, description="Normalized Difference Vegetation Index.")
    elevation: float = Field(..., ge=-500, le=9000, description="Elevation in meters above sea level.")

    # Advanced wildfire training features. Defaults preserve older clients that
    # only submit the original six demo fields, while the frontend now sends all
    # values explicitly for trained-model inference.
    latitude: float = Field(0.0, ge=-90, le=90, description="Latitude in decimal degrees.")
    longitude: float = Field(0.0, ge=-180, le=180, description="Longitude in decimal degrees.")
    pressure_mean: float = Field(1013.25, ge=800, le=1100, description="Mean atmospheric pressure in hPa.")
    solar_radiation_mean: float = Field(250.0, ge=0, le=1400, description="Mean solar radiation in W/m2.")
    evapotranspiration_total: float = Field(5.0, ge=0, le=30, description="Total evapotranspiration in mm.")
    cloud_cover_mean: float = Field(20.0, ge=0, le=100, description="Mean cloud cover percentage.")
    dewpoint_mean: float = Field(10.0, ge=-80, le=50, description="Mean dew point in degrees Celsius.")
    wind_direction_mean: float = Field(180.0, ge=0, le=360, description="Mean wind direction in degrees.")

class PredictionResponse(BaseModel):
    risk_probability: float
    intensity: float
    risk_label: str
    confidence: float
    message: str

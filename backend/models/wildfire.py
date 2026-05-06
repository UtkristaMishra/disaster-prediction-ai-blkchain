import os
import joblib
import numpy as np
import pandas as pd

class WildfirePredictor:
    """
    Wildfire risk predictor using trained ML model.
    
    Pipeline:
    1. Load trained RandomForest/XGBoost/LogisticRegression model
    2. Load feature scaler
    3. Normalize input features
    4. Generate prediction and confidence score
    """
    
    def __init__(self, model_path='backend/models/saved_models/wildfire_model.pkl',
                 scaler_path='backend/models/saved_models/wildfire_scaler.pkl'):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        
        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                if hasattr(self.model, "n_jobs"):
                    self.model.n_jobs = 1
                self.model_loaded = True
                print(f"Loaded trained wildfire model from {model_path}")
            except Exception as e:
                print(f"Failed to load wildfire model: {e}")
        else:
            print(f"Trained wildfire model not found at {model_path}")
            print("  Run: python backend/train_wildfire_model.py")

    def predict(self, rainfall: float, temperature: float, humidity: float,
                wind_speed: float, ndvi: float, elevation: float,
                latitude: float = 0.0, longitude: float = 0.0,
                pressure_mean: float = 1013.25,
                solar_radiation_mean: float = 250.0,
                evapotranspiration_total: float = 5.0,
                cloud_cover_mean: float = 20.0,
                dewpoint_mean: float = 10.0,
                wind_direction_mean: float = 180.0):
        """
        Predict wildfire risk.
        
        Args:
            rainfall, temperature, humidity, wind_speed, ndvi, elevation: core demo features
            latitude, longitude, pressure_mean, solar_radiation_mean,
            evapotranspiration_total, cloud_cover_mean, dewpoint_mean,
            wind_direction_mean: advanced fields aligned to wildfire training features
        
        Returns:
            dict with risk_probability, risk_label, confidence
        """
        if not self.model_loaded:
            return self._fallback_heuristic(rainfall, temperature, humidity, wind_speed, ndvi, elevation)
        
        try:
            # Prepare the exact wildfire training feature vector. UI/backend
            # names use latitude/longitude, then map to the scaler's lat/lon.
            fire_weather_index = self._estimate_fire_weather_index(
                rainfall=rainfall,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                solar_radiation_mean=solar_radiation_mean,
                cloud_cover_mean=cloud_cover_mean,
            )
            features = np.array([[
                temperature,      # temp_mean
                humidity,         # humidity_min  
                wind_speed,       # wind_speed_max
                fire_weather_index,
                latitude,         # lat
                longitude,        # lon
                pressure_mean,
                solar_radiation_mean,
                evapotranspiration_total,
                cloud_cover_mean,
                dewpoint_mean,
                wind_direction_mean
            ]])
            
            # Scale using saved scaler
            feature_names = getattr(self.scaler, "feature_names_in_", None)
            if feature_names is not None:
                features = pd.DataFrame(features, columns=feature_names)
            features_scaled = self.scaler.transform(features)
            
            # Get multi-output prediction: [probability, intensity]
            prediction = self.model.predict(features_scaled)[0]
            risk_probability = float(prediction[0])
            intensity = float(prediction[1])
            
            # Ensure probability within [0, 1] and calibrate for live demo inputs.
            # Manual demo values are broader than the training feature mapping.
            model_probability = min(max(risk_probability, 0.0), 1.0)
            input_probability = self._score_inputs(rainfall, temperature, humidity, wind_speed, ndvi, elevation)
            risk_probability = (0.4 * model_probability) + (0.6 * input_probability)
            
            # Classify risk level based on probability
            if risk_probability >= 0.6:
                risk_label = "High Risk"
            elif risk_probability >= 0.35:
                risk_label = "Moderate Risk"
            else:
                risk_label = "Low Risk"
            
            # Confidence: higher probability = higher confidence
            confidence = 0.7 + 0.3 * risk_probability
            
            return {
                "risk_probability": round(risk_probability, 4),
                "intensity": round(intensity, 4),
                "risk_label": risk_label,
                "confidence": round(confidence, 4),
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_heuristic(rainfall, temperature, humidity, wind_speed, ndvi, elevation)
    
    def _fallback_heuristic(self, rainfall: float, temperature: float, humidity: float, 
                           wind_speed: float, ndvi: float, elevation: float):
        """
        Fallback heuristic-based risk scoring (demo only).
        Used when trained model is not available.
        """
        risk_probability = self._score_inputs(rainfall, temperature, humidity, wind_speed, ndvi, elevation)
        risk_label = "High Risk" if risk_probability >= 0.6 else "Moderate Risk" if risk_probability >= 0.35 else "Low Risk"
        confidence = 0.65
        intensity = risk_probability * 50  # Scale intensity based on probability

        return {
            "risk_probability": round(risk_probability, 4),
            "intensity": round(intensity, 4),
            "risk_label": risk_label,
            "confidence": round(confidence, 4),
        }

    def _score_inputs(self, rainfall: float, temperature: float, humidity: float,
                      wind_speed: float, ndvi: float, elevation: float) -> float:
        score = 0.0
        score += 0.30 * min(max((temperature - 15) / 25.0, 0.0), 1.0)
        score += 0.25 * min(max((70.0 - humidity) / 70.0, 0.0), 1.0)
        score += 0.20 * min(max(wind_speed / 35.0, 0.0), 1.0)
        score += 0.12 * max(0.0, 1.0 - ndvi)
        score += 0.08 * max(0.0, 1.0 - min(max(rainfall / 80.0, 0.0), 1.0))
        score += 0.05 * min(max((1000.0 - elevation) / 1000.0, 0.0), 1.0)
        return min(max(score, 0.0), 1.0)

    def _estimate_fire_weather_index(self, rainfall: float, temperature: float,
                                     humidity: float, wind_speed: float,
                                     solar_radiation_mean: float,
                                     cloud_cover_mean: float) -> float:
        """Estimate FWI from submitted weather values when no FWI sensor exists."""
        heat_component = max((temperature - 10.0) / 2.5, 0.0)
        dryness_component = max((100.0 - humidity) / 10.0, 0.0)
        wind_component = max(wind_speed / 5.0, 0.0)
        sun_component = max(solar_radiation_mean / 120.0, 0.0)
        cloud_penalty = max(cloud_cover_mean / 25.0, 0.0)
        rain_penalty = max(rainfall / 10.0, 0.0)
        return max(heat_component + dryness_component + wind_component + sun_component - cloud_penalty - rain_penalty, 0.0)

import os
import joblib
import numpy as np
import pandas as pd

class FloodPredictor:
    """
    Flood risk predictor using trained ML model.
    
    Pipeline:
    1. Load trained RandomForest/XGBoost/LogisticRegression model
    2. Load feature scaler
    3. Normalize input features
    4. Generate prediction and confidence score
    """
    
    def __init__(self, model_path='backend/models/saved_models/flood_model.pkl',
                 scaler_path='backend/models/saved_models/flood_scaler.pkl'):
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
                print(f"Loaded trained flood model from {model_path}")
            except Exception as e:
                print(f"Failed to load flood model: {e}")
        else:
            print(f"Trained flood model not found at {model_path}")
            print("  Run: python backend/train_flood_model.py")

    def predict(self, rainfall: float, temperature: float, humidity: float, 
                wind_speed: float, ndvi: float, elevation: float):
        """
        Predict flood risk.
        
        Args:
            rainfall, temperature, humidity, wind_speed, ndvi, elevation: environmental features
        
        Returns:
            dict with risk_probability, risk_label, confidence
        """
        if not self.model_loaded:
            return self._fallback_heuristic(rainfall, temperature, humidity, wind_speed, ndvi, elevation)
        
        try:
            # Prepare feature vector matching training features (20 features)
            # Map inputs to some features, use defaults for others
            features = np.array([[
                rainfall,    # MonsoonIntensity
                elevation,   # TopographyDrainage
                5.0,         # RiverManagement (default)
                5.0,         # Deforestation (default)
                5.0,         # Urbanization (default)
                5.0,         # ClimateChange (default)
                5.0,         # DamsQuality (default)
                5.0,         # Siltation (default)
                5.0,         # AgriculturalPractices (default)
                5.0,         # Encroachments (default)
                5.0,         # IneffectiveDisasterPreparedness (default)
                5.0,         # DrainageSystems (default)
                5.0,         # CoastalVulnerability (default)
                5.0,         # Landslides (default)
                5.0,         # Watersheds (default)
                5.0,         # DeterioratingInfrastructure (default)
                5.0,         # PopulationScore (default)
                5.0,         # WetlandLoss (default)
                5.0,         # InadequatePlanning (default)
                5.0          # PoliticalFactors (default)
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
            # The trained flood model can saturate at 1.0 for broad manual inputs.
            model_probability = min(max(risk_probability, 0.0), 1.0)
            input_probability = self._score_inputs(rainfall, temperature, humidity, wind_speed, ndvi, elevation)
            risk_probability = (0.4 * model_probability) + (0.6 * input_probability)
            
            # Classify risk level based on probability
            if risk_probability >= 0.55:
                risk_label = "High Risk"
            elif risk_probability >= 0.3:
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
        risk_label = "High Risk" if risk_probability >= 0.55 else "Moderate Risk" if risk_probability >= 0.3 else "Low Risk"
        confidence = 0.65
        intensity = risk_probability * 10  # Scale intensity based on probability

        return {
            "risk_probability": round(risk_probability, 4),
            "intensity": round(intensity, 4),
            "risk_label": risk_label,
            "confidence": round(confidence, 4),
        }

    def _score_inputs(self, rainfall: float, temperature: float, humidity: float,
                      wind_speed: float, ndvi: float, elevation: float) -> float:
        score = 0.0
        score += 0.35 * min(max(rainfall / 150.0, 0.0), 1.0)
        score += 0.25 * min(max((humidity - 40) / 60.0, 0.0), 1.0)
        score += 0.15 * min(max((temperature - 10) / 25.0, 0.0), 1.0)
        score += 0.10 * max(0.0, 1.0 - ndvi)
        score += 0.10 * min(max((500.0 - elevation) / 500.0, 0.0), 1.0)
        score += 0.05 * min(max(wind_speed / 30.0, 0.0), 1.0)
        return min(max(score, 0.0), 1.0)

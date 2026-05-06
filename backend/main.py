from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.predictor import ModelRegistry
from backend.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Disaster Prediction Prototype",
    description="Modular FastAPI backend for wildfire and flood prediction models.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ModelRegistry()

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

@app.post("/predict/wildfire", response_model=PredictionResponse)
def predict_wildfire(request: PredictionRequest):
    try:
        # Use the full validated payload so wildfire inference receives the
        # same environmental feature space that was used during training.
        result = predictor.predict(
            model_name="wildfire",
            **request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return PredictionResponse(
        risk_probability=result["risk_probability"],
        intensity=result["intensity"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        message="Wildfire prediction completed successfully",
    )

@app.post("/predict/flood", response_model=PredictionResponse)
def predict_flood(request: PredictionRequest):
    try:
        # Flood currently uses the original six demo features and ignores
        # wildfire-specific advanced fields inside its predictor.
        result = predictor.predict(
            model_name="flood",
            **request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return PredictionResponse(
        risk_probability=result["risk_probability"],
        intensity=result["intensity"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        message="Flood prediction completed successfully",
    )

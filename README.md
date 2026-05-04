# Disaster Prediction Model Using AI, Federated Learning, Blockchain, and IPFS

## Project Overview

This is a college IPD prototype for a decentralized disaster risk prediction platform. It combines a baseline AI/ML pipeline with a lightweight FastAPI backend, a simple blockchain audit contract, and a demo frontend.

The current implementation focuses on:
- baseline model and preprocessing
- FastAPI backend for prediction requests
- blockchain smart contract prototype for immutable logging
- IPFS concept for off-chain artifact storage
- beginner-friendly modular architecture

## Current Folder Structure

- `backend/`: FastAPI app and prediction logic
- `contracts/`: Solidity smart contract for audit logging
- `frontend/`: minimal HTML/CSS/JS demo UI
- `notebooks/`: starter ML notebook prototype
- `data/`: placeholder for CSV datasets

## Quick Start

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and place datasets:
   - See `data/README.md` for dataset download instructions
   - Place `wildfire_dataset.csv` and `flood_dataset.csv` in `data/` folder
4. Train models (run once):
   ```bash
   python backend/train_wildfire_model.py
   python backend/train_flood_model.py
   ```
   This trains Random Forest, XGBoost, and Logistic Regression models and saves the best performers.
5. Start the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```
6. Open `frontend/index.html` in a browser to test predictions using the trained models.

## ML Pipeline Architecture

The implementation uses a two-stage approach:

### Stage 1: Model Training (One-time)
- Located in: `backend/train_wildfire_model.py` and `backend/train_flood_model.py`
- Loads real CSV datasets from `data/` folder
- Trains three baseline models:
  - Random Forest Classifier
  - XGBoost Classifier
  - Logistic Regression
- Evaluates using: accuracy, precision, recall, F1 score
- Saves best model + scaler using joblib to `backend/models/saved_models/`

### Stage 2: Real-time Prediction (FastAPI Backend)
- Located in: `backend/models/wildfire.py` and `backend/models/flood.py`
- Loads trained models on startup
- API endpoints normalize input features using saved scaler
- Returns predictions with confidence scores
- Graceful fallback to heuristic logic if models not trained yet

## API Endpoints

- `GET /health` — backend health check
- `POST /predict/wildfire` — wildfire risk prediction (uses trained ML model)
- `POST /predict/flood` — flood risk prediction (uses trained ML model)
## Dataset Guidance

A better dataset for the current prototype is a real wildfire CSV from Kaggle, such as a California wildfire dataset with roughly 20,000–30,000 rows. This stays within the recommended 5,000–50,000 row range and avoids heavy imagery or GPU processing.

Important candidate features:
- rainfall
- temperature
- humidity
- wind_speed
- ndvi or vegetation index
- elevation

## Next Prototype Steps

The project will be developed incrementally:
1. add tabular dataset and preprocessing pipeline
2. implement a baseline ML model training flow
3. connect the backend to the trained model
4. add IPFS hashing and logging to blockchain contract
5. improve the frontend demo for predictions

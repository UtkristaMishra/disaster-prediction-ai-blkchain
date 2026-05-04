"""
Wildfire model training script.

This script trains baseline ML models (Random Forest, XGBoost, Linear Regression)
on the wildfire dataset and saves the best-performing model.

Run once to train models:
    python backend/train_wildfire_model.py

Models and scalers are saved to backend/models/saved_models/
"""

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Will train Random Forest and Linear Regression only.")

from backend.preprocessing import load_wildfire_data


def train_wildfire_models():
    """
    Train wildfire prediction models.
    """
    print("="*60)
    print("WILDFIRE MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n[1/3] Loading data...")
    X_train, X_test, y_train, y_test, scaler = load_wildfire_data()
    
    # Create output directory
    model_dir = 'backend/models/saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    results = {}
    
    # Train Random Forest
    print("\n[2/3] Training models...")
    print("\n--- Random Forest ---")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    print(f"MSE:       {rf_mse:.4f}")
    print(f"R2 Score:  {rf_r2:.4f}")
    results['RandomForest'] = rf_r2
    
    # Train XGBoost if available
    if HAS_XGBOOST:
        print("\n--- XGBoost ---")
        xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        
        xgb_mse = mean_squared_error(y_test, y_pred_xgb)
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        
        print(f"MSE:       {xgb_mse:.4f}")
        print(f"R2 Score:  {xgb_r2:.4f}")
        results['XGBoost'] = xgb_r2
    
    # Train Linear Regression
    print("\n--- Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    lr_mse = mean_squared_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    print(f"MSE:       {lr_mse:.4f}")
    print(f"R2 Score:  {lr_r2:.4f}")
    results['LinearRegression'] = lr_r2
    
    # Select best model
    print("\n[3/3] Saving best model...")
    best_model_name = max(results, key=results.get)
    best_r2 = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} (R2: {best_r2:.4f})")
    
    # Save best model and scaler
    if best_model_name == 'RandomForest':
        best_model = rf_model
    elif best_model_name == 'XGBoost':
        best_model = xgb_model
    else:
        best_model = lr_model
    
    joblib.dump(best_model, f'{model_dir}/wildfire_model.pkl')
    joblib.dump(scaler, f'{model_dir}/wildfire_scaler.pkl')
    
    print(f"Wildfire model saved to: {model_dir}/wildfire_model.pkl")
    print(f"Wildfire scaler saved to: {model_dir}/wildfire_scaler.pkl")
    print("\n" + "="*60)


if __name__ == '__main__':
    train_wildfire_models()

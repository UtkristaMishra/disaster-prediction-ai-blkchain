"""
Data preprocessing and loading module for wildfire and flood prediction.

This module handles:
- Loading CSV datasets from the data/ folder
- Feature selection and normalization
- Train/test split
- Data validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Define feature mappings for each model
WILDFIRE_FEATURES = [
    'temp_mean',
    'humidity_min',
    'wind_speed_max',
    'fire_weather_index',
    'lat',
    'lon',
    'pressure_mean',
    'solar_radiation_mean',
    'evapotranspiration_total',
    'cloud_cover_mean',
    'dewpoint_mean',
    'wind_direction_mean',
]

WILDFIRE_TARGET = 'occured'

WILDFIRE_TARGETS = ['occured', 'frp']

FLOOD_FEATURES = [
    'MonsoonIntensity',
    'TopographyDrainage',
    'RiverManagement',
    'Deforestation',
    'Urbanization',
    'ClimateChange',
    'DamsQuality',
    'Siltation',
    'AgriculturalPractices',
    'Encroachments',
    'IneffectiveDisasterPreparedness',
    'DrainageSystems',
    'CoastalVulnerability',
    'Landslides',
    'Watersheds',
    'DeterioratingInfrastructure',
    'PopulationScore',
    'WetlandLoss',
    'InadequatePlanning',
    'PoliticalFactors',
]

FLOOD_TARGET = 'FloodProbability'

FLOOD_TARGETS = ['FloodProbability', 'MonsoonIntensity']


def load_wildfire_data(data_path='data/wildfire_dataset.csv', test_size=0.2, random_state=42):
    """
    Load and preprocess wildfire dataset.
    
    Args:
        data_path: path to wildfire CSV file
        test_size: fraction for test set
        random_state: for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please download from Kaggle.")
    
    print(f"Loading wildfire dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Select only available features
    available_features = [f for f in WILDFIRE_FEATURES if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    X = df[available_features].fillna(0)
    y = df[WILDFIRE_TARGETS].fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_flood_data(data_path='data/flood_dataset.csv', test_size=0.2, random_state=42):
    """
    Load and preprocess flood dataset.
    
    Args:
        data_path: path to flood CSV file
        test_size: fraction for test set
        random_state: for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please download from Kaggle.")
    
    print(f"Loading flood dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Select only available features
    available_features = [f for f in FLOOD_FEATURES if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    X = df[available_features].fillna(0)
    y = df[FLOOD_TARGETS].fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

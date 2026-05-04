# Data Folder

## Recommended Real Datasets

This folder stores prototype datasets for training wildfire and flood prediction models.

### 1. Wildfire Prediction Dataset

**Dataset:** GLOBAL WILDFIRE DATASET  
**Source:** [Kaggle](https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction)  
**File:** `wildfire_dataset.csv`  
**Rows:** 118,858 (can be downsampled to ~50k for prototype)  
**Columns:** 17 numeric features  

**Key features:**
- `temp_mean` (temperature)
- `humidity_min` (humidity)
- `wind_speed_max` (wind speed)
- `fire_weather_index` (fire risk indicator)
- `frp` (Fire Radiative Power)
- `occurred` (binary fire label: 0/1)

**Download instructions:**
1. Visit https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction
2. Click "Download" and extract `final_dataset.csv`
3. Rename to `wildfire_dataset.csv` and place in this folder
4. (Optional: sample to 50k rows with `pandas.read_csv(...).sample(50000).to_csv(...)` for faster training)

---

### 2. Flood Prediction Dataset

**Dataset:** Flood Prediction Dataset  
**Source:** [Kaggle](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset)  
**File:** `flood_dataset.csv`  
**Rows:** 50,000  
**Columns:** 21 numeric features  

**Key features:**
- `Rainfall_mm` (rainfall)
- `Temperature_C` (temperature)
- `Humidity_pct` (humidity)
- `DrainageSystems` (drainage quality)
- `RiverManagement` (river management effectiveness)
- `FloodProbability` (continuous target, 0–1)

**Download instructions:**
1. Visit https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset
2. Click "Download" and extract `flood.csv`
3. Rename to `flood_dataset.csv` and place in this folder

---

## Dataset Size & CPU Performance

- Wildfire: ~118k rows (22 MB) → downsample to 50k for student laptop
- Flood: 50k rows (2.35 MB) → use as-is
- Both: tabular CSV format, no image/raster processing, CPU-friendly
- Estimated training time: <5 minutes per model on modern laptop

---

## Next Steps

Once datasets are in this folder:
1. The ML notebook will load them automatically
2. Backend predictors can integrate real trained models
3. Preprocessing pipeline can normalize features

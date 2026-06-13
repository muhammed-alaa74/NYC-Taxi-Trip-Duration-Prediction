# NYC Taxi Trip Duration Prediction

A machine learning project that predicts the duration of NYC taxi trips using
pickup/dropoff coordinates, datetime, and passenger information. The final
model is served through an interactive Streamlit application with a
map-based interface for selecting pickup and dropoff points.

## Live Demo

[Add your Streamlit Cloud link here once deployed]
<!-- Example: https://nyc-taxi-trip-duration.streamlit.app -->

## Problem Statement

Given pickup/dropoff coordinates, pickup datetime, and passenger info,
predict how long a taxi trip will take (in seconds).

## Project Structure

```
nyc-taxi-trip-duration/
├── app/
│   └── app.py                  # Streamlit application (interactive map UI)
├── src/
│   ├── feature_engineering.py  # Datetime features and haversine distance
│   ├── preprocessing.py        # Categorical encoding and train/test split
│   ├── train.py                # XGBoost training and artifact saving
│   ├── evaluate.py              # Model evaluation utilities (MAE, RMSE, R2)
│   ├── predict.py              # Model loading, feature alignment, prediction
│   └── utils.py                # Haversine distance helper
├── notebooks/
│   └── nyc-taxi.ipynb          # Exploratory analysis and model comparison
├── models/
│   ├── xgb_model.pkl           # Trained XGBoost model
│   └── features.pkl            # Feature column order used at training time
├── datasets/
│   └── nyc_taxi_trip_duration.csv
├── outputs/                     # EDA plots and evaluation result charts
├── main.py                      # Full training pipeline (CLI)
├── pipeline_test.py             # End-to-end prediction pipeline test
├── requirements.txt
└── README.md
```

## Dataset

[NYC Taxi Trip Duration - Kaggle](https://www.kaggle.com/datasets/parisrohan/nyc-taxi-trip-duration)

## Feature Engineering

| Feature | Description |
|---|---|
| `hour`, `day`, `month`, `weekday` | Extracted from `pickup_datetime` |
| `is_weekend` | 1 if Saturday or Sunday |
| `distance` | Haversine distance (km) between pickup and dropoff |
| `store_and_fwd_flag` | Encoded N to 0, Y to 1 |

The target variable `trip_duration` is log-transformed (`log1p`) during
training and converted back (`expm1`) for evaluation and prediction.

## Models

Several ensemble models were trained and compared during exploration
(see `notebooks/nyc-taxi.ipynb`):

| Model | Notes |
|---|---|
| XGBoost | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` |
| Random Forest | `n_estimators=100`, `max_depth=15` |
| LightGBM | `n_estimators=300`, `learning_rate=0.05` |
| Extra Trees | `n_estimators=300`, `max_depth=15` |
| AdaBoost | `n_estimators=200`, `learning_rate=0.05` |
| HistGradientBoosting | `max_iter=500`, `learning_rate=0.05` |
| CatBoost | Default parameters |

XGBoost was selected as the production model (R2 approximately 0.80 on the
held-out test set) and is the model used by `src/train.py`, `src/predict.py`,
and the Streamlit app.

## Evaluation Metrics

Models are evaluated on the original (seconds) scale using:

- MAE - Mean Absolute Error
- RMSE - Root Mean Squared Error
- R2 - Coefficient of determination

## Pipeline Overview

1. `src/feature_engineering.py` parses `pickup_datetime` and computes
   time-based features and haversine distance.
2. `src/preprocessing.py` drops unused columns (`id`, `dropoff_datetime`)
   and encodes `store_and_fwd_flag`.
3. `src/train.py` trains the XGBoost model on the log-transformed target
   and saves `models/xgb_model.pkl` and `models/features.pkl`.
4. `src/predict.py` loads the saved model and feature list, aligns incoming
   data to the expected columns, and returns predictions in seconds.

## Installation and Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python main.py --data datasets/nyc_taxi_trip_duration.csv
```

Run EDA plots alongside training:

```bash
python main.py --data datasets/nyc_taxi_trip_duration.csv --eda
```

Run the Streamlit app locally:

```bash
streamlit run app/app.py
```

Load the saved model directly:

```python
import joblib
import numpy as np

model = joblib.load("models/xgb_model.pkl")
predictions_log = model.predict(X_test)
predictions_seconds = np.expm1(predictions_log)
```

## Application Features

- Interactive map (Folium) bounded to New York City for selecting pickup
  and dropoff points
- Sidebar controls for vendor, passenger count, pickup date, and time
- Real-time route visualization between pickup and dropoff
- Predicted trip duration shown in seconds, minutes, and hours

## Tech Stack

- Python
- scikit-learn, XGBoost, LightGBM, CatBoost
- Streamlit, Folium, streamlit-folium
- pandas, numpy, joblib

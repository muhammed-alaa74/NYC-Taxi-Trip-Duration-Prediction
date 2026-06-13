# NYC Taxi Trip Duration Prediction

A complete, end-to-end machine learning project that predicts how long a New
York City taxi trip will take, based on pickup and dropoff coordinates,
pickup time, passenger count, and a few engineered features. The project
covers data exploration, cleaning, feature engineering, training and
comparing multiple regression models, and serving the best model through an
interactive, map-based Streamlit web application.

## Table of Contents

- [Live Demo](#live-demo)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Exploratory Data Analysis and Data Cleaning](#exploratory-data-analysis-and-data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Models and Training](#models-and-training)
- [Results](#results)
- [Pipeline Architecture](#pipeline-architecture)
- [The Streamlit Application](#the-streamlit-application)
- [Installation and Local Setup](#installation-and-local-setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Tech Stack](#tech-stack)
- [Dataset Credit](#dataset-credit)

## Live Demo

[[Live Web Application](https://nyc-taxi-trip-duration-prediction-mcfyenyegjyzfkskg9ppbw.streamlit.app)]

## Overview

Given a pickup location, a dropoff location, the pickup date and time, the
number of passengers, and whether the trip record was stored and forwarded
(`store_and_fwd_flag`), the goal is to predict the trip duration in seconds.

The project is split into two parts:

1. A research and experimentation phase (`notebooks/nyc-taxi.ipynb`) where the
   raw dataset is explored, cleaned, engineered, and used to train and compare
   seven different regression models plus a weighted ensemble.
2. A production phase (`src/` and `app/`) where the best-performing model
   (XGBoost) is trained through a reusable pipeline, saved as a `.pkl`
   artifact, and served through a Streamlit application with an interactive
   map.

## Project Structure

```
nyc-taxi-trip-duration/
├── app/
│   └── app.py                  # Streamlit application (interactive map UI)
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py  # Datetime features and haversine distance
│   ├── preprocessing.py        # Categorical encoding and train/test split
│   ├── train.py                # XGBoost training and artifact saving
│   ├── evaluate.py             # Model evaluation utilities (MAE, RMSE, R2)
│   ├── predict.py              # Model loading, feature alignment, prediction
│   └── utils.py                # Haversine distance helper
├── notebooks/
│   └── nyc-taxi.ipynb          # Exploratory analysis and model comparison
├── models/
│   ├── xgb_model.pkl           # Trained XGBoost model (production model)
│   └── features.pkl            # Feature column order used at training time
├── datasets/
│   └── nyc_taxi_trip_duration.csv
├── outputs/                     # EDA plots and evaluation result charts
├── main.py                       # Full training pipeline (CLI)
├── pipeline_test.py              # End-to-end prediction pipeline test
├── requirements.txt
└── README.md
```

## Dataset

The project uses the [NYC Taxi Trip Duration dataset from Kaggle](https://www.kaggle.com/datasets/parisrohan/nyc-taxi-trip-duration).

Original size: 729,322 rows and 11 columns.

| Column | Description |
|---|---|
| `id` | Unique trip identifier (dropped before training) |
| `vendor_id` | Taxi vendor / dispatch company (1 or 2) |
| `pickup_datetime` | Date and time the trip started |
| `dropoff_datetime` | Date and time the trip ended (dropped before training) |
| `passenger_count` | Number of passengers |
| `pickup_longitude`, `pickup_latitude` | Pickup coordinates |
| `dropoff_longitude`, `dropoff_latitude` | Dropoff coordinates |
| `store_and_fwd_flag` | Whether the trip record was held in vehicle memory before sending (Y/N) |
| `trip_duration` | Target variable, in seconds |

## Exploratory Data Analysis and Data Cleaning

The notebook (`notebooks/nyc-taxi.ipynb`) performs the following steps before
any model is trained:

1. **Datetime decomposition.** `pickup_datetime` is converted to a proper
   datetime type and used to extract `hour`, `day`, `month`, `weekday`, and
   `is_weekend`.

2. **Haversine distance.** A `distance` column (in kilometers) is computed
   from the pickup and dropoff coordinates using the haversine formula, which
   measures great-circle distance between two latitude/longitude pairs.

3. **Categorical encoding.** `store_and_fwd_flag` is mapped from `N`/`Y` to
   `0`/`1`.

4. **Target transformation.** `trip_duration` ranges from 1 second to over
   1,939,736 seconds (about 22 days), which is heavily right-skewed. A
   `trip_duration_log = log1p(trip_duration)` column is created and used as
   the actual training target. Predictions are converted back to seconds
   with `expm1` at evaluation and inference time.

5. **Geographic outlier removal.** Some coordinates in the raw data fall far
   outside New York City (for example, latitude as low as 32 or as high as
   51, and longitude as far west as -121). The dataset is filtered to a
   bounding box around NYC:
   - Pickup latitude: 40.5 to 41.0
   - Pickup longitude: -74.05 to -73.7
   - Dropoff latitude: 40.5 to 41.0
   - Dropoff longitude: -74.05 to -73.7

6. **Trip duration outlier removal.** Trips of 60 seconds or less and trips of
   12,600 seconds (3.5 hours) or more are removed, as these are likely data
   errors, cancelled trips, or extreme outliers. This removes 5,223 rows and
   leaves 721,811 rows with a more reasonable distribution (mean about 837
   seconds, max 12,419 seconds).

7. **Passenger count cleaning.** Trips with 0 passengers (10 rows) or 9
   passengers (1 row, almost certainly a data entry error) are removed. The
   dataset is restricted to 1 to 6 passengers.

8. **Correlation analysis.** A correlation matrix is computed between
   `trip_duration` and the engineered features. `distance` has the strongest
   correlation with trip duration (about 0.77), followed by pickup longitude.
   Time-based features (`hour`, `day`, `month`, `weekday`, `is_weekend`) have
   weak individual correlations but still contribute useful signal when
   combined in a non-linear model such as XGBoost.

All of the EDA visualizations referenced above (box plots of trip duration
before and after the log transform, histograms, correlation heatmaps, and bar
charts of trip duration by vendor and passenger count) are saved as images in
the `outputs/` folder.

## Feature Engineering

After cleaning, the following features are used to train the model:

| Feature | Description |
|---|---|
| `vendor_id` | Taxi vendor (1 or 2) |
| `passenger_count` | Number of passengers (1 to 6) |
| `pickup_longitude`, `pickup_latitude` | Pickup coordinates |
| `dropoff_longitude`, `dropoff_latitude` | Dropoff coordinates |
| `store_and_fwd_flag` | Encoded as 0 (N) or 1 (Y) |
| `hour`, `day`, `month`, `weekday` | Extracted from `pickup_datetime` |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |
| `distance` | Haversine distance (km) between pickup and dropoff |

The target is `trip_duration_log = log1p(trip_duration)`.

`pickup_datetime`, `dropoff_datetime`, `id`, `trip_duration`, and
`trip_duration_log` are dropped from the feature matrix before training (see
`src/preprocessing.py`).

## Models and Training

The notebook trains and compares seven regression models, all using the
log-transformed target, with an 80/20 train/test split:

| Model | Hyperparameters |
|---|---|
| XGBoost | `n_estimators=2000`, `learning_rate=0.03`, `max_depth=6`, `tree_method=hist` |
| Random Forest | `n_estimators=100`, `max_depth=15` |
| LightGBM | `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31` |
| Extra Trees | `n_estimators=300`, `max_depth=15` |
| AdaBoost | `n_estimators=200`, `learning_rate=0.05` |
| HistGradientBoosting | `max_iter=2000`, `learning_rate=0.03`, `max_depth=10` |
| CatBoost | `iterations=2000`, `learning_rate=0.03`, `depth=6` |

In addition, a weighted ensemble combining the two strongest models
(70% XGBoost + 30% HistGradientBoosting predictions, averaged in log space)
is evaluated.

XGBoost was chosen as the production model because it achieved the best
individual R2 score and the ensemble offered no meaningful improvement over
it. Using a single model also keeps the production pipeline and the
Streamlit app simpler and faster.

`src/train.py` retrains this XGBoost configuration on the full cleaned
dataset and saves two artifacts to `models/`:

- `xgb_model.pkl` - the trained XGBoost regressor
- `features.pkl` - the ordered list of feature column names expected by the
  model, used by `src/predict.py` to align incoming data at inference time

## Results

All metrics below are computed on the held-out test set (20% of the cleaned
data), after converting predictions back to seconds with `expm1`.

| Model | MAE (seconds) | RMSE (seconds) | R2 |
|---|---|---|---|
| XGBoost | 171.54 | 288.35 | 0.8034 |
| HistGradientBoosting | 174.15 | 291.86 | 0.7985 |
| Ensemble (70% XGBoost + 30% HistGradientBoosting) | 171.56 | 288.50 | 0.8032 |
| CatBoost | 180.27 | 300.51 | 0.7864 |
| LightGBM | 187.37 | 310.36 | 0.7722 |
| Random Forest | 191.64 | 314.25 | 0.7665 |
| Extra Trees | 204.63 | 328.70 | 0.7445 |
| AdaBoost | 258.29 | 411.40 | 0.5997 |

**Interpretation.** An R2 of about 0.80 means the model explains roughly 80%
of the variance in trip duration. An MAE of about 171 seconds (just under
3 minutes) means that, on average, predictions are off by about 3 minutes
from the actual trip duration. Boosting-based tree models (XGBoost,
HistGradientBoosting, CatBoost) clearly outperform bagging-based models
(Random Forest, Extra Trees) and AdaBoost on this dataset, which is expected
given the non-linear relationships between distance, time of day, and trip
duration.

## Pipeline Architecture

The production pipeline mirrors the steps used during experimentation, but is
split into small, reusable modules under `src/`:

1. **`src/feature_engineering.py`**
   - Parses `pickup_datetime` into a proper datetime
   - Extracts `hour`, `day`, `month`, `weekday`, `is_weekend`
   - Computes `distance` using `src/utils.py`'s haversine implementation
   - If `trip_duration` is present (training only), also computes
     `trip_duration_log`

2. **`src/preprocessing.py`**
   - `encode_categorical`: drops `id` and `dropoff_datetime` if present, and
     encodes `store_and_fwd_flag` to 0/1
   - `prepare_splits`: drops datetime and target columns, and returns an
     80/20 train/test split (`X_train`, `X_test`, `y_train`, `y_test`)

3. **`src/train.py`**
   - Loads `datasets/nyc_taxi_trip_duration.csv`
   - Runs feature engineering and encoding
   - Splits the data and trains the XGBoost model described above
   - Saves `models/xgb_model.pkl` and `models/features.pkl`

4. **`src/predict.py`**
   - `load_model` / `load_features`: load the saved artifacts
   - `align_features`: reindexes an incoming DataFrame to match the exact
     column order the model was trained on, filling any missing columns
     with 0
   - `predict`: runs the model and converts the log-scale prediction back to
     seconds with `expm1`

5. **`main.py`** ties everything together as a command-line entry point, with
   an optional `--eda` flag to regenerate exploratory plots.

6. **`pipeline_test.py`** and **`test_src_pipeline.py`** run a single sample
   trip through the full pipeline (feature engineering, encoding, alignment,
   prediction) to verify that everything works end to end before deployment.

## The Streamlit Application

`app/app.py` is a Streamlit application that lets a user interactively pick a
pickup point and a dropoff point on a map of New York City, configure trip
parameters, and get a predicted trip duration.

### Layout

The page is split into two columns:

- **Left column**: a large interactive map, plus dynamic instructions that
  guide the user through the current step.
- **Right column**: a results panel showing the selected pickup and dropoff
  coordinates, the chosen number of passengers, a "Run Model Prediction"
  button, and (after prediction) the predicted trip duration.

The sidebar contains the trip configuration controls:

- **Vendor Dispatcher**: choice between Vendor 1 (Creative Mobile
  Technologies) and Vendor 2 (VeriFone Inc.)
- **Passenger Count**: 1 to 6
- **Pickup Date** and **Pickup Time**
- **Map Mode Selection**: a radio button that determines whether the next
  click on the map sets the pickup location or the dropoff location
- **Clear & Reset Map** button, which clears both selected points

### How the map interaction works

1. The map is built with Folium (`build_master_map`), centered on New York
   City and constrained to a bounding box so users cannot pan or zoom far
   outside the city.

2. The app keeps track of three pieces of state in
   `st.session_state`:
   - `pickup_coords`
   - `dropoff_coords`
   - `selection_mode` (either `"pickup"` or `"dropoff"`)

3. When the user clicks on the map, `st_folium` returns the clicked
   latitude/longitude. Depending on `selection_mode`, the app stores this
   click as either `pickup_coords` or `dropoff_coords` and triggers a rerun
   (`st.rerun()`) so the map redraws immediately with the new marker.

4. As points are set, the map updates in real time:
   - A green marker is added for the pickup point
   - A red marker is added for the dropoff point
   - Once both points are set, a blue line is drawn between them and the map
     automatically zooms to fit the route

5. The instruction banner above the map changes dynamically:
   - "Click anywhere on the map to set your Pickup Location" when no pickup
     point is set yet
   - "Now change the mode on the sidebar to 'Set Dropoff Location' and click
     your destination" once the pickup point is set
   - "Your route is ready, click Run Model Prediction" once both points are
     set

### Running a prediction

When the user clicks **Run Model Prediction**:

1. The app validates that both pickup and dropoff points are set (and shows
   a warning message if not).
2. It builds a single-row DataFrame from the selected pickup/dropoff
   coordinates, the chosen date and time (combined into `pickup_datetime`),
   passenger count, vendor, and a default `store_and_fwd_flag` of `"N"`.
3. This row is passed through the same `feature_engineering` and
   `encode_categorical` functions used in training, then aligned to the
   model's expected feature columns with `align_features`.
4. `predict` runs the XGBoost model and converts the result back to seconds.
5. The result panel shows the predicted duration in minutes (as the headline
   number), plus seconds and hours in two smaller cards underneath.

### Caching

The model and feature list are loaded once and cached with
`@st.cache_resource`, so the `.pkl` files are not reloaded from disk on every
interaction.

## Installation and Local Setup

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the model

Run the full training pipeline from the project root:

```bash
python main.py --data datasets/nyc_taxi_trip_duration.csv
```

Add `--eda` to also regenerate the exploratory data analysis plots:

```bash
python main.py --data datasets/nyc_taxi_trip_duration.csv --eda
```

This will produce (or overwrite) `models/xgb_model.pkl` and
`models/features.pkl`.

### Running the prediction pipeline test

To verify the trained model and feature alignment work end to end on a single
sample trip:

```bash
python pipeline_test.py
```

### Running the Streamlit app locally

```bash
streamlit run app/app.py
```

This opens the app in your browser, typically at `http://localhost:8501`.

### Loading the model directly in Python

```python
import joblib
import numpy as np

model = joblib.load("models/xgb_model.pkl")
features = joblib.load("models/features.pkl")

predictions_log = model.predict(X_test[features])
predictions_seconds = np.expm1(predictions_log)
```

## Deployment

This project is deployed using Streamlit Community Cloud.

- **Repository:** `NYC-Taxi-Trip-Duration-Prediction`
- **Branch:** `main`
- **Main file:** `app/app.py`

### Live Demo

[[Live Web Application](https://nyc-taxi-trip-duration-prediction-mcfyenyegjyzfkskg9ppbw.streamlit.app)]

### requirements.txt

The app depends on `scikit-learn` even though it is not imported directly in
`app.py`, because `src/preprocessing.py` imports
`from sklearn.model_selection import train_test_split` at module level.
Make sure `scikit-learn` is included in `requirements.txt`, otherwise the
Streamlit Cloud build will fail with a `ModuleNotFoundError`.

## Limitations and Future Improvements

- The model is trained only on trips with pickup and dropoff points inside
  the NYC bounding box used during cleaning. Predictions for points far
  outside this area (for example, airports outside the bounding box, or other
  cities) may be unreliable.
- Trip duration is capped between 1 minute and 3.5 hours in the training
  data, so the model may underestimate unusually long trips.
- The Streamlit app currently uses a fixed `store_and_fwd_flag = "N"` and
  does not account for traffic, weather, or special events.
- Potential improvements:
  - Add weather and traffic data as additional features
  - Experiment with neighborhood/zone-based features instead of raw
    coordinates
  - Add prediction intervals (uncertainty estimates) alongside the point
    estimate
  - Re-introduce the weighted ensemble (XGBoost + HistGradientBoosting) in
    the app if the small accuracy gain is worth the extra inference cost

## Tech Stack

- **Language**: Python
- **Modeling**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data handling**: pandas, numpy
- **App / UI**: Streamlit, Folium, streamlit-folium
- **Serialization**: joblib

## Dataset Credit

[NYC Taxi Trip Duration - Kaggle](https://www.kaggle.com/datasets/parisrohan/nyc-taxi-trip-duration)

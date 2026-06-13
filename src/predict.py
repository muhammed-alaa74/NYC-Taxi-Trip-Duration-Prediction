from pathlib import Path
import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def load_features():
    return joblib.load(FEATURES_PATH)


def align_features(df, features):
    return df.reindex(columns=features, fill_value=0)


def predict(model, df):
    pred_log = model.predict(df)
    return np.expm1(pred_log)
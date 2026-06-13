import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor

from src.feature_engineering import feature_engineering
from src.preprocessing import encode_categorical, prepare_splits


def train_model(X_train, y_train):

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def save_artifacts(model, X_train):

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model, models_dir / "xgb_model.pkl")
    joblib.dump(list(X_train.columns), models_dir / "features.pkl")


def main():

    print("Loading data...")
    base_dir = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base_dir / "datasets" / "nyc_taxi_trip_duration.csv")

    print("Feature engineering...")
    df = feature_engineering(df)

    print("Encoding...")
    df = encode_categorical(df)

    print("Splitting...")
    X_train, X_test, y_train, y_test = prepare_splits(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Saving artifacts...")
    save_artifacts(model, X_train)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
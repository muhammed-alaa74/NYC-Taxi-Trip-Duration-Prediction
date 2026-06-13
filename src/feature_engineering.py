import numpy as np
import pandas as pd
from src.utils import haversine_distance


def feature_engineering(df):

    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )

    if "trip_duration" in df.columns:
        df["trip_duration_log"] = np.log1p(df["trip_duration"])

    return df
import pandas as pd

from src.feature_engineering import feature_engineering
from src.preprocessing import encode_categorical
from src.predict import load_model, load_features, align_features, predict


sample = pd.DataFrame([{
    "pickup_datetime": "2016-03-25 15:30:00",
    "pickup_latitude": 40.7589,
    "pickup_longitude": -73.9851,
    "dropoff_latitude": 40.6413,
    "dropoff_longitude": -73.7781,
    "passenger_count": 1,
    "store_and_fwd_flag": "N"
}])


# PIPELINE
df = feature_engineering(sample)
df = encode_categorical(df)

model = load_model()
features = load_features()

df = align_features(df, features)

result = predict(model, df)

print("Prediction:", result[0])
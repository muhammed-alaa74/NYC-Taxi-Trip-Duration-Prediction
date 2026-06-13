import pandas as pd

from src.feature_engineering import feature_engineering
from src.preprocessing import encode_categorical
from src.predict import load_model, load_features, align_features, predict


print("Loading sample...")

# 1) Sample input (زي ما اليوزر هيدخل في Streamlit)
sample = pd.DataFrame([{
    "pickup_datetime": "2016-03-25 15:30:00",
    "pickup_latitude": 40.7589,
    "pickup_longitude": -73.9851,
    "dropoff_latitude": 40.6413,
    "dropoff_longitude": -73.7781,
    "passenger_count": 1,
    "store_and_fwd_flag": "N"
}])


# 2) Feature engineering
print("Feature engineering...")
df = feature_engineering(sample)

# 3) Encoding
print("Encoding...")
df = encode_categorical(df)

# 4) Load model + features
print("Loading model...")
model = load_model()
features = load_features()

# 5) Align features
print("Aligning features...")
df = align_features(df, features)

# 6) Predict
print("Predicting...")
result = predict(model, df)

print("\n====================")
print("✅ SUCCESS")
print("====================")
print("Prediction (seconds):", result[0])
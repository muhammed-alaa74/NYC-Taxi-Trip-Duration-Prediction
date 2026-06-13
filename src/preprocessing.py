from sklearn.model_selection import train_test_split


def encode_categorical(df):

    df = df.copy()

    # حذف الكولومز اللي مش محتاجينها
    drop_cols = ["id", "dropoff_datetime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1})

    return df


def prepare_splits(df, target_col="trip_duration_log"):

    drop_cols = [
        "pickup_datetime",
        "dropoff_datetime",
        "trip_duration",
        "trip_duration_log"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)
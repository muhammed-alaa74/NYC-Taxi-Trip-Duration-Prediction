import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, model_name, X_test, y_test):

    pred_log = model.predict(X_test)

    preds = np.expm1(pred_log)

    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, preds)

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            preds
        )
    )

    r2 = r2_score(
        y_true,
        preds
    )

    print(
        f"{model_name:<10} | "
        f"MAE: {mae:.2f} | "
        f"RMSE: {rmse:.2f} | "
        f"R2: {r2:.4f}"
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_all_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        results[name] = evaluate_model(
            model,
            name.upper(),
            X_test,
            y_test
        )

    return results
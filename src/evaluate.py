from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np
import json

def evaluate_model(
        model,
        X_test,
        y_test):

    preds = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    mse = mean_squared_error(
        y_test,
        preds
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        preds
    )

    metrics = {

        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    with open(
        "artifacts/metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return metrics
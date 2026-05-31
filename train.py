import pandas as pd
import joblib
import json

from src.data_preprocessing import preprocess
from src.model_trainer import train_model
from src.evaluate import evaluate_model

def train_pipeline():

    df = pd.read_csv(
        "data/housing.csv"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor

    ) = preprocess(
        df,
        "median_house_value"
    )

    model = train_model(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    joblib.dump(
        model,
        "model.pkl"
    )

    joblib.dump(
        preprocessor,
        "preprocessor.pkl"
    )

    with open(
        "metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return model, preprocessor


if __name__ == "__main__":
    train_pipeline()
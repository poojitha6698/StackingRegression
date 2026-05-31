import joblib
import pandas as pd

model = joblib.load(
    "artifacts/model.pkl"
)

preprocessor = joblib.load(
    "artifacts/preprocessor.pkl"
)

def predict(input_df):

    transformed = preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        transformed
    )

    return prediction[0]
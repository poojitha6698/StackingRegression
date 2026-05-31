import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Stacking Regression",
    page_icon="🏠",
    layout="wide"
)

# Auto-train if model doesn't exist

if (
    not os.path.exists("model.pkl")
    or
    not os.path.exists("preprocessor.pkl")
):

    from train import train_pipeline

    with st.spinner(
        "Training model for first deployment..."
    ):
        train_pipeline()

model = joblib.load(
    "model.pkl"
)

preprocessor = joblib.load(
    "preprocessor.pkl"
)

st.title(
    "🏠 California Housing Price Prediction"
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    data = pd.read_csv(
        uploaded_file
    )

    st.subheader(
        "Input Dataset"
    )

    st.dataframe(
        data.head()
    )

    transformed = preprocessor.transform(
        data
    )

    predictions = model.predict(
        transformed
    )

    data[
        "Predicted_House_Value"
    ] = predictions

    st.subheader(
        "Predictions"
    )

    st.dataframe(
        data.head()
    )

    csv = data.to_csv(
        index=False
    )

    st.download_button(
        "Download Results",
        csv,
        "predictions.csv",
        "text/csv"
    )
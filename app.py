import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

model = joblib.load(
    "artifacts/model.pkl"
)

preprocessor = joblib.load(
    "artifacts/preprocessor.pkl"
)

st.title("🏠 California Housing Price Prediction")

st.markdown("""
Predict California House Prices using

✅ Random Forest

✅ Gradient Boosting

✅ Stacking Regressor
""")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    data = pd.read_csv(
        uploaded_file
    )

    st.subheader("Uploaded Data")

    st.dataframe(data.head())

    transformed = preprocessor.transform(
        data
    )

    predictions = model.predict(
        transformed
    )

    data["Predicted_House_Value"] = predictions

    st.subheader("Predictions")

    st.dataframe(
        data.head()
    )

    csv = data.to_csv(
        index=False
    )

    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )
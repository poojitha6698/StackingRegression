import streamlit as st

import pandas as pd

import joblib

import os

# ==========================================
# IMPORT EDA FUNCTIONS
# ==========================================

from src.eda import (

    charges_distribution,
    bmi_vs_charges,
    age_vs_charges
)

# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = 'models/stacking_regression_model.pkl'

# ==========================================
# TRAIN MODEL IF MODEL DOES NOT EXIST
# ==========================================

if not os.path.exists(MODEL_PATH):

    st.warning(
        'Model not found. Training model...'
    )

    import models.train_model

# ==========================================
# LOAD MODEL SAFELY
# ==========================================

try:

    model = joblib.load(
        MODEL_PATH
    )

except Exception as e:

    st.warning(
        'Model incompatible or corrupted.'
    )

    st.warning(
        'Retraining model automatically...'
    )

    import models.train_model

    model = joblib.load(
        MODEL_PATH
    )

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    'insurance.csv'
)

# ==========================================
# STREAMLIT PAGE TITLE
# ==========================================

st.title(
    'Insurance Charges Prediction using Stacking Regression'
)

# ==========================================
# DATASET PREVIEW
# ==========================================

st.header(
    'Dataset Preview'
)

st.dataframe(
    df.head()
)

# ==========================================
# EDA SECTION
# ==========================================

st.header(
    'Exploratory Data Analysis'
)

# ------------------------------------------
# Charges Distribution
# ------------------------------------------

fig1 = charges_distribution(df)

st.pyplot(fig1)

# ------------------------------------------
# BMI vs Charges
# ------------------------------------------

fig2 = bmi_vs_charges(df)

st.pyplot(fig2)

# ------------------------------------------
# Age vs Charges
# ------------------------------------------

fig3 = age_vs_charges(df)

st.pyplot(fig3)

# ==========================================
# USER INPUT SECTION
# ==========================================

st.header(
    'Enter User Details'
)

# ------------------------------------------
# Age
# ------------------------------------------

age = st.slider(

    'Age',

    18,

    100,

    25
)

# ------------------------------------------
# Sex
# ------------------------------------------

sex = st.selectbox(

    'Sex',

    [
        'male',
        'female'
    ]
)

# ------------------------------------------
# BMI
# ------------------------------------------

bmi = st.slider(

    'BMI',

    10.0,

    50.0,

    25.0
)

# ------------------------------------------
# Children
# ------------------------------------------

children = st.slider(

    'Children',

    0,

    5,

    0
)

# ------------------------------------------
# Smoker
# ------------------------------------------

smoker = st.selectbox(

    'Smoker',

    [
        'yes',
        'no'
    ]
)

# ------------------------------------------
# Region
# ------------------------------------------

region = st.selectbox(

    'Region',

    [
        'southwest',
        'southeast',
        'northwest',
        'northeast'
    ]
)

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({

    'age': [age],

    'sex': [sex],

    'bmi': [bmi],

    'children': [children],

    'smoker': [smoker],

    'region': [region],

    'bmi_age_interaction': [bmi * age]
})

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    'Predict Insurance Charges'
):

    prediction = model.predict(
        input_data
    )[0]

    st.success(

        f'Predicted Insurance Charges : ${prediction:.2f}'
    )

# ==========================================
# FOOTER
# ==========================================

st.write(
    'Stacking Regression Project using Streamlit'
)
import streamlit as st

import pandas as pd

import joblib

from src.eda import (

    charges_distribution,
    bmi_vs_charges,
    age_vs_charges
)

# Title

st.title(
    'Insurance Charges Prediction using Stacking Regression'
)

# Load Dataset

df = pd.read_csv(
    'insurance.csv'
)

# Load Model

model = joblib.load(
    'models/stacking_regression_model.pkl'
)

# Dataset Preview

st.header(
    'Dataset Preview'
)

st.dataframe(
    df.head()
)

# EDA Section

st.header(
    'Exploratory Data Analysis'
)

# Charges Distribution

fig1 = charges_distribution(df)

st.pyplot(fig1)

# BMI vs Charges

fig2 = bmi_vs_charges(df)

st.pyplot(fig2)

# Age vs Charges

fig3 = age_vs_charges(df)

st.pyplot(fig3)

# User Inputs

st.header(
    'Enter User Details'
)

age = st.slider(
    'Age',
    18,
    100,
    25
)

sex = st.selectbox(
    'Sex',
    ['male', 'female']
)

bmi = st.slider(
    'BMI',
    10.0,
    50.0,
    25.0
)

children = st.slider(
    'Children',
    0,
    5,
    0
)

smoker = st.selectbox(
    'Smoker',
    ['yes', 'no']
)

region = st.selectbox(
    'Region',
    [
        'southwest',
        'southeast',
        'northwest',
        'northeast'
    ]
)

# Input Data

input_data = pd.DataFrame({

    'age': [age],

    'sex': [sex],

    'bmi': [bmi],

    'children': [children],

    'smoker': [smoker],

    'region': [region],

    'bmi_age_interaction': [bmi * age]
})

# Prediction

if st.button(
    'Predict Charges'
):

    prediction = model.predict(
        input_data
    )[0]

    st.success(

        f'Predicted Insurance Charges : ${prediction:.2f}'
    )


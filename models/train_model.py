import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.ensemble import (

    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression

from src.load_data import load_dataset

from src.preprocess import create_preprocessor

from src.feature_engineering import perform_feature_engineering

# ======================
# LOAD DATA
# ======================

df = load_dataset(
    'insurance.csv'
)

# ======================
# FEATURE ENGINEERING
# ======================

df = perform_feature_engineering(df)

# ======================
# FEATURES AND TARGET
# ======================

X = df.drop(
    'charges',
    axis=1
)

y = df['charges']

# ======================
# TRAIN TEST SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# ======================
# PREPROCESSOR
# ======================

preprocessor = create_preprocessor()

# ======================
# BASE LEARNERS
# ======================

base_learners = [

    (
        'rf',

        RandomForestRegressor(

            n_estimators=100,

            random_state=42
        )
    ),

    (
        'gbr',

        GradientBoostingRegressor(

            n_estimators=100,

            random_state=42
        )
    ),

    (
        'etr',

        ExtraTreesRegressor(

            n_estimators=100,

            random_state=42
        )
    )
]

# ======================
# META LEARNER
# ======================

meta_learner = LinearRegression()

# ======================
# STACKING MODEL
# ======================

stacking_model = StackingRegressor(

    estimators=base_learners,

    final_estimator=meta_learner,

    cv=5
)

# ======================
# PIPELINE
# ======================

pipeline = Pipeline(

    steps=[

        (
            'preprocessor',
            preprocessor
        ),

        (
            'model',
            stacking_model
        )
    ]
)

# ======================
# TRAIN MODEL
# ======================

pipeline.fit(
    X_train,
    y_train
)

# ======================
# SAVE MODEL
# ======================

joblib.dump(

    pipeline,

    'models/stacking_regression_model.pkl'
)

print(
    'Model Saved Successfully'
)
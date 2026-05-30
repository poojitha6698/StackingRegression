import sys
import os

# Add project root directory to Python path

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

import joblib

from sklearn.model_selection import (
    train_test_split
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.ensemble import (

    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)

from sklearn.linear_model import (
    LinearRegression
)

from src.load_data import (
    load_dataset
)

from src.preprocess import (
    create_preprocessor
)

from src.feature_engineering import (
    perform_feature_engineering
)

from utils.helper import (
    evaluate_model
)

# =========================
# LOAD DATASET
# =========================

df = load_dataset(
    'insurance.csv'
)

print('\nDataset Loaded Successfully\n')

# =========================
# FEATURE ENGINEERING
# =========================

df = perform_feature_engineering(df)

print('Feature Engineering Completed\n')

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop(
    'charges',
    axis=1
)

y = df['charges']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

print('Train Test Split Completed\n')

# =========================
# PREPROCESSOR
# =========================

preprocessor = create_preprocessor()

print('Preprocessor Created\n')

# =========================
# BASE LEARNERS
# =========================

base_learners = [

    (
        'random_forest',

        RandomForestRegressor(

            n_estimators=100,

            max_depth=10,

            random_state=42
        )
    ),

    (
        'gradient_boosting',

        GradientBoostingRegressor(

            n_estimators=100,

            random_state=42
        )
    ),

    (
        'extra_trees',

        ExtraTreesRegressor(

            n_estimators=100,

            random_state=42
        )
    )
]

# =========================
# META LEARNER
# =========================

meta_learner = LinearRegression()

# =========================
# STACKING REGRESSOR
# =========================

stacking_model = StackingRegressor(

    estimators=base_learners,

    final_estimator=meta_learner,

    cv=5
)

print('Stacking Regressor Created\n')

# =========================
# PIPELINE
# =========================

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

# =========================
# TRAIN MODEL
# =========================

print('Training Model...\n')

pipeline.fit(
    X_train,
    y_train
)

print('Model Training Completed\n')

# =========================
# PREDICTIONS
# =========================

y_pred = pipeline.predict(
    X_test
)

print('Prediction Completed\n')

# =========================
# EVALUATION
# =========================

mae, mse, rmse, r2 = evaluate_model(

    y_test,
    y_pred
)

print('==============================')
print('MODEL EVALUATION METRICS')
print('==============================\n')

print('MAE : ', mae)

print('MSE : ', mse)

print('RMSE : ', rmse)

print('R2 SCORE : ', r2)

# =========================
# SAVE MODEL
# =========================

joblib.dump(

    pipeline,

    'models/stacking_regression_model.pkl'
)

print('\nModel Saved Successfully\n')
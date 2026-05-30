import sys
import os

# ==========================================
# ADD PROJECT ROOT DIRECTORY TO PYTHON PATH
# ==========================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

# ==========================================
# IMPORT LIBRARIES
# ==========================================

from sklearn.model_selection import (
    train_test_split
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.metrics import (

    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import (

    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
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

# ==========================================
# LOAD DATASET
# ==========================================

df = load_dataset(
    'insurance.csv'
)

print('\nDataset Loaded Successfully\n')

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df = perform_feature_engineering(df)

print('Feature Engineering Completed\n')

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(
    'charges',
    axis=1
)

y = df['charges']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

print('Train Test Split Completed\n')

# ==========================================
# CREATE PREPROCESSOR
# ==========================================

preprocessor = create_preprocessor()

# ==========================================
# INDIVIDUAL MODELS
# ==========================================

models = {

    'Random Forest':

    RandomForestRegressor(

        n_estimators=100,

        random_state=42
    ),

    'Gradient Boosting':

    GradientBoostingRegressor(

        n_estimators=100,

        random_state=42
    ),

    'Extra Trees':

    ExtraTreesRegressor(

        n_estimators=100,

        random_state=42
    )
}

# ==========================================
# MODEL EVALUATION
# ==========================================

print('=================================')
print('INDIVIDUAL MODEL PERFORMANCE')
print('=================================\n')

for name, model in models.items():

    pipeline = Pipeline(

        steps=[

            (
                'preprocessor',
                preprocessor
            ),

            (
                'model',
                model
            )
        ]
    )

    # ======================================
    # TRAIN MODEL
    # ======================================

    pipeline.fit(
        X_train,
        y_train
    )

    # ======================================
    # PREDICTIONS
    # ======================================

    predictions = pipeline.predict(
        X_test
    )

    # ======================================
    # METRICS
    # ======================================

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    # ======================================
    # PRINT RESULTS
    # ======================================

    print(f'{name} Results\n')

    print('R2 Score : ', r2)

    print('MAE : ', mae)

    print('MSE : ', mse)

    print('RMSE : ', rmse)

    print('-----------------------------\n')
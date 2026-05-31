from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train):

    estimators = [

        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        ),

        (
            "gb",
            GradientBoostingRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]

    stacking_model = StackingRegressor(

        estimators=estimators,

        final_estimator=LinearRegression(),

        cv=5,

        n_jobs=-1
    )

    stacking_model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        stacking_model,
        "artifacts/model.pkl"
    )

    return stacking_model
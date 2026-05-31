from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):

    estimators = [

        (
            "rf",
            RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        ),

        (
            "gb",
            GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=3,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model
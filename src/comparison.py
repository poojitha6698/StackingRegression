from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import r2_score

def compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        stacking_model):

    models = {

        "Linear Regression":
        LinearRegression(),

        "Random Forest":
        RandomForestRegressor(),

        "Gradient Boosting":
        GradientBoostingRegressor(),

        "Stacking":
        stacking_model
    }

    results = {}

    for name, model in models.items():

        if name != "Stacking":

            model.fit(
                X_train,
                y_train
            )

        pred = model.predict(X_test)

        score = r2_score(
            y_test,
            pred
        )

        results[name] = score

    return results
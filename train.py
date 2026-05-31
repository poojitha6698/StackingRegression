from src.data_ingestion import load_data
from src.data_preprocessing import preprocess
from src.model_trainer import train_model
from src.evaluate import evaluate_model
from src.comparison import compare_models

def main():

    df = load_data(
        "data/housing.csv"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor

    ) = preprocess(
        df,
        "median_house_value"
    )

    model = train_model(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    comparison = compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        model
    )

    print("\nMetrics")
    print(metrics)

    print("\nModel Comparison")
    print(comparison)

if __name__ == "__main__":
    main()
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

def create_preprocessor():

    categorical_features = [
        'sex',
        'smoker',
        'region'
    ]

    numerical_features = [
        'age',
        'bmi',
        'children',
        'bmi_age_interaction'
    ]

    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore'
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[

            (
                'categorical',
                categorical_transformer,
                categorical_features
            ),

            (
                'numerical',
                numerical_transformer,
                numerical_features
            )
        ]
    )

    return preprocessor
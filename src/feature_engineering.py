def perform_feature_engineering(df):

    df[
        'bmi_age_interaction'
    ] = df['bmi'] * df['age']

    return df
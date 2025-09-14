import pandas as pd
import json


def load_feature_config(featureConfigPath: str):
    with open(featureConfigPath, 'r') as f:
        featureConfig = json.load(f)
        return featureConfig


def preprocess_data(csvPath: str, featureConfigPath: str):
    df = pd.read_csv(csvPath)
    feature_config = load_feature_config(featureConfigPath)

    numerical_cols = [col for col, config in feature_config.items()
                      if config['type'] == 'numerical']
    categorical_cols = [col for col, config in feature_config.items()
                        if config['type'] == 'categorical']

    for col in numerical_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        vocabulary = feature_config[col]['vocabulary']

        df_encoded = pd.get_dummies(df[col], prefix=col)

        for category in vocabulary:
            col_name = f'{col}_{category}'
            if col_name not in df_encoded.columns:
                df_encoded[col_name] = 0

        df = pd.concat([df.drop(columns=[col]), df_encoded], axis=1)

    return df
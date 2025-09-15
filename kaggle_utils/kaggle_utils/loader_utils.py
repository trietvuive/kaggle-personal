import pandas as pd
import json


def load_feature_config(feature_config_path: str):
    with open(feature_config_path, 'r') as f:
        feature_config = json.load(f)
        return feature_config


def preprocess_data(csv_path: str,
                    feature_config_path: str,
                    columns_to_excluded: list[str] = []):
    df = pd.read_csv(csv_path)
    feature_config = load_feature_config(feature_config_path)

    for col in columns_to_excluded:
        feature_config.pop(col, 'not_found')

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

    return df.drop(columns_to_excluded, axis=1, errors='ignore')
"""
Data service module for loading and preprocessing the sleep debt dataset.

This module provides common data loading and preprocessing utilities
shared across all classification model implementations.
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Path to the dataset relative to the project root
DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "bedtime_screentime_sleep_debt.csv",
)

# Target column
TARGET_COLUMN = "sleep_debt_category"

# Columns to drop (non-predictive identifiers)
COLUMNS_TO_DROP = ["user_id"]

# Columns which are predictive but not classification data
COLUMNS_TO_DROP.append("next_day_fatigue_score")

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    "gender",
    "chronotype",
    "occupation_type",
    "primary_bedtime_app",
    "blue_light_filter_active",
]


def load_data(file_path: str = DATA_FILE) -> pd.DataFrame:
    """
    Load the raw CSV dataset into a pandas DataFrame.

    Args:
        file_path: Absolute path to the CSV file. Defaults to the project data file.

    Returns:
        Raw DataFrame with all columns.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
) -> dict:
    """
    Preprocess the DataFrame and split it into train/test sets.

    Steps performed:
        1. Trim whitespace, remove empty or incomplete rows, and discard rows
           with invalid numeric feature values.
        2. Drop non-predictive columns (e.g. user_id).
        3. Separate features and target, then split into training and test sets.
        4. Fit one-hot encoding and the optional numeric scaler on training
           data only.
        5. Transform both sets using the training-fitted preprocessing.

    Args:
        df:            Raw DataFrame returned by load_data().
        test_size:     Fraction of data reserved for testing (default 0.2).
        random_state:  Seed for reproducible splits (default 42).
        scale_features: Whether to apply StandardScaler to features (default True).

    Returns:
        A dictionary with the following keys:
            - "X_train": Training feature matrix (numpy array).
            - "X_test":  Test feature matrix (numpy array).
            - "y_train": Training labels (numpy array).
            - "y_test":  Test labels (numpy array).
            - "feature_names": Raw input feature column names.
            - "model_feature_names": Encoded feature names used by the estimator.
            - "categorical_encoder": One-hot encoder fitted on training features.
            - "label_encoder": Fitted LabelEncoder for the target column.
            - "scaler": Numeric-feature scaler, or None if scaling is disabled.
    """
    dataset = df.copy()
    for column in dataset.select_dtypes(include=["object", "string"]).columns:
        dataset[column] = dataset[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    dataset = dataset.replace(r"^\s*$", pd.NA, regex=True)
    numeric_columns = [
        column
        for column in dataset.columns
        if column not in CATEGORICAL_COLUMNS
           and column not in {TARGET_COLUMN, "user_id"}
    ]
    for column in numeric_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    if numeric_columns:
        valid_numeric_rows = np.isfinite(
            dataset[numeric_columns].to_numpy(dtype=float)
        ).all(axis=1)
        dataset = dataset.loc[valid_numeric_rows]
    dataset = dataset.dropna(axis=0, how="any")
    if dataset.empty:
        raise ValueError(
            "No rows remain after removing empty, incomplete, or invalid rows."
        )

    dataset = dataset.drop(columns=COLUMNS_TO_DROP, errors="ignore").copy()
    feature_frame = dataset.drop(columns=[TARGET_COLUMN])
    target = dataset[TARGET_COLUMN]
    feature_names = feature_frame.columns.tolist()
    categorical_features = [
        column for column in CATEGORICAL_COLUMNS if column in feature_names
    ]
    numeric_features = [
        column for column in feature_names if column not in categorical_features
    ]

    # Encode target labels before splitting so the split can be stratified.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_encoder = None
    if categorical_features:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        categorical_encoder.fit(X_train[categorical_features])
        X_train_categorical = categorical_encoder.transform(
            X_train[categorical_features]
        )
        X_test_categorical = categorical_encoder.transform(
            X_test[categorical_features]
        )
        encoded_categorical_names = (
            categorical_encoder.get_feature_names_out(categorical_features).tolist()
        )
    else:
        X_train_categorical = np.empty((len(X_train), 0))
        X_test_categorical = np.empty((len(X_test), 0))
        encoded_categorical_names = []

    X_train_numeric = X_train[numeric_features].to_numpy(dtype=float)
    X_test_numeric = X_test[numeric_features].to_numpy(dtype=float)
    scaler = None
    if scale_features and numeric_features:
        scaler = StandardScaler()
        X_train_numeric = scaler.fit_transform(X_train_numeric)
        X_test_numeric = scaler.transform(X_test_numeric)

    X_train_array = np.hstack((X_train_numeric, X_train_categorical))
    X_test_array = np.hstack((X_test_numeric, X_test_categorical))
    model_feature_names = numeric_features + encoded_categorical_names

    return {
        "X_train": X_train_array,
        "X_test": X_test_array,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "model_feature_names": model_feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_encoder": categorical_encoder,
        "label_encoder": label_encoder,
        "scaler": scaler,
    }


class FittedModelArtifact:
    """Adapt an implementation estimator to accept raw feature rows."""

    def __init__(
            self,
            estimator: Any,
            feature_names: list[str],
            numeric_features: list[str],
            categorical_features: list[str],
            categorical_encoder: OneHotEncoder | None,
            scaler: StandardScaler | None,
    ) -> None:
        self.estimator = estimator
        self.feature_names = feature_names
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.categorical_encoder = categorical_encoder
        self.scaler = scaler

    def predict(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            feature_frame = features.loc[:, self.feature_names].copy()
        else:
            feature_array = np.asarray(features)
            if feature_array.ndim != 2 or feature_array.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} input features."
                )
            feature_frame = pd.DataFrame(feature_array, columns=self.feature_names)

        for column in feature_frame.select_dtypes(
            include=["object", "string"]
        ).columns:
            feature_frame[column] = feature_frame[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        numeric_values = feature_frame[self.numeric_features].to_numpy(dtype=float)
        if self.scaler is not None:
            numeric_values = self.scaler.transform(numeric_values)
        if self.categorical_encoder is not None:
            categorical_values = self.categorical_encoder.transform(
                feature_frame[self.categorical_features]
            )
        else:
            categorical_values = np.empty((len(feature_frame), 0))
        values = np.hstack((numeric_values, categorical_values))
        return self.estimator.predict(values)


def get_preprocessed_data(
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
        file_path: str = DATA_FILE,
) -> dict:
    """
    Convenience function that loads and preprocesses the dataset in one call.

    Args:
        test_size:      Fraction of data reserved for testing (default 0.2).
        random_state:   Seed for reproducible splits (default 42).
        scale_features: Whether to apply StandardScaler to features (default True).
        file_path:      CSV file to load (defaults to the project dataset).

    Returns:
        Preprocessed data dictionary (same structure as preprocess_data()).
    """
    df = load_data(file_path)
    return preprocess_data(df, test_size=test_size, random_state=random_state, scale_features=scale_features)

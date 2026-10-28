"""
Data service module for loading and preprocessing the sleep debt dataset.

This module provides common data loading and preprocessing utilities
shared across all classification model implementations.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
COLUMNS_TO_DROP.extend("next_day_fatigue_score")

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
        1. Drop non-predictive columns (e.g. user_id).
        2. Encode categorical features with LabelEncoder.
        3. Separate features (X) from the target (y).
        4. Encode the target labels.
        5. Optionally scale numerical features with StandardScaler.
        6. Split into training and test sets.

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
            - "feature_names": List of feature column names.
            - "label_encoder": Fitted LabelEncoder for the target column.
            - "scaler": Fitted StandardScaler (or None if scale_features=False).
    """
    df = df.copy()

    # 1. Drop non-predictive columns
    df.drop(columns=COLUMNS_TO_DROP, errors="ignore", inplace=True)

    # 2. Encode categorical features
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 3. Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = X.columns.tolist()

    # 4. Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 5. Optionally scale features
    scaler = None
    X_array = X.values
    if scale_features:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "scaler": scaler,
    }


def get_preprocessed_data(
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> dict:
    """
    Convenience function that loads and preprocesses the dataset in one call.

    Args:
        test_size:      Fraction of data reserved for testing (default 0.2).
        random_state:   Seed for reproducible splits (default 42).
        scale_features: Whether to apply StandardScaler to features (default True).

    Returns:
        Preprocessed data dictionary (same structure as preprocess_data()).
    """
    df = load_data()
    return preprocess_data(df, test_size=test_size, random_state=random_state, scale_features=scale_features)

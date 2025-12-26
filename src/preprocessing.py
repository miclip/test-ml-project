"""
Data Preprocessing Pipeline for Customer Churn Prediction

This module provides functions for preprocessing customer data,
including missing value handling, categorical encoding, feature scaling,
and train/validation/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


# Default split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random state for reproducibility
RANDOM_STATE = 42


def identify_column_types(
    df: pd.DataFrame,
    target_column: str = 'churn',
    id_column: str = 'customer_id'
) -> Dict[str, List[str]]:
    """
    Identify numerical and categorical columns in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column to exclude.
    id_column : str
        Name of the ID column to exclude.

    Returns
    -------
    dict
        Dictionary with 'numerical' and 'categorical' column lists.
    """
    exclude_cols = [target_column, id_column]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    numerical_cols = []
    categorical_cols = []

    for col in feature_cols:
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


def handle_missing_values(
    df: pd.DataFrame,
    numerical_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent'
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential missing values.
    numerical_strategy : str
        Strategy for numerical columns: 'mean', 'median', or 'most_frequent'.
    categorical_strategy : str
        Strategy for categorical columns: 'most_frequent' or 'constant'.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()
    col_types = identify_column_types(df)

    # Handle numerical columns
    if col_types['numerical']:
        num_imputer = SimpleImputer(strategy=numerical_strategy)
        df[col_types['numerical']] = num_imputer.fit_transform(
            df[col_types['numerical']]
        )

    # Handle categorical columns
    if col_types['categorical']:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        df[col_types['categorical']] = cat_imputer.fit_transform(
            df[col_types['categorical']]
        )

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    encoding_type: str = 'onehot',
    drop_first: bool = True
) -> pd.DataFrame:
    """
    Encode categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_columns : list of str, optional
        Columns to encode. If None, auto-detect.
    encoding_type : str
        Type of encoding: 'onehot' or 'label'.
    drop_first : bool
        For one-hot encoding, whether to drop first category.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded categorical features.
    """
    df = df.copy()

    if categorical_columns is None:
        col_types = identify_column_types(df)
        categorical_columns = col_types['categorical']

    if not categorical_columns:
        return df

    if encoding_type == 'onehot':
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=drop_first,
            dtype=int
        )
    elif encoding_type == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def scale_numerical_features(
    df: pd.DataFrame,
    numerical_columns: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numerical_columns : list of str, optional
        Columns to scale. If None, auto-detect.
    scaler : StandardScaler, optional
        Pre-fitted scaler to use. If None, fit a new one.

    Returns
    -------
    tuple
        (Scaled DataFrame, fitted scaler)
    """
    df = df.copy()

    if numerical_columns is None:
        col_types = identify_column_types(df)
        numerical_columns = col_types['numerical']

    if not numerical_columns:
        return df, StandardScaler()

    if scaler is None:
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        df[numerical_columns] = scaler.transform(df[numerical_columns])

    return df, scaler


def create_train_val_test_split(
    df: pd.DataFrame,
    target_column: str = 'churn',
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column for stratification.
    train_ratio : float
        Proportion for training set (default: 0.70).
    val_ratio : float
        Proportion for validation set (default: 0.15).
    test_ratio : float
        Proportion for test set (default: 0.15).
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        Whether to stratify by target column.

    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"

    stratify_col = df[target_column] if stratify else None

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=stratify_col
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df[target_column] if stratify else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        random_state=random_state,
        stratify=stratify_temp
    )

    return train_df, val_df, test_df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = 'churn',
    id_column: str = 'customer_id'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column.
    id_column : str
        Name of the ID column to exclude.

    Returns
    -------
    tuple
        (X features DataFrame, y target Series)
    """
    drop_cols = [target_column]
    if id_column in df.columns:
        drop_cols.append(id_column)

    X = df.drop(columns=drop_cols)
    y = df[target_column]

    return X, y


class ChurnPreprocessor:
    """
    Complete preprocessing pipeline for churn prediction.

    This class encapsulates all preprocessing steps and can be saved/loaded
    for reproducible inference.
    """

    def __init__(
        self,
        numerical_impute_strategy: str = 'median',
        categorical_impute_strategy: str = 'most_frequent',
        encoding_type: str = 'onehot',
        scale_features: bool = True,
        random_state: int = RANDOM_STATE
    ):
        self.numerical_impute_strategy = numerical_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.encoding_type = encoding_type
        self.scale_features = scale_features
        self.random_state = random_state

        self.scaler = None
        self.feature_columns = None
        self.is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str = 'churn'
    ) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        target_column : str
            Name of target column.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.
        """
        df = df.copy()

        # Handle missing values
        df = handle_missing_values(
            df,
            numerical_strategy=self.numerical_impute_strategy,
            categorical_strategy=self.categorical_impute_strategy
        )

        # Encode categorical features
        df = encode_categorical_features(
            df,
            encoding_type=self.encoding_type
        )

        # Scale numerical features
        if self.scale_features:
            col_types = identify_column_types(df, target_column=target_column)
            df, self.scaler = scale_numerical_features(
                df,
                numerical_columns=col_types['numerical']
            )

        self.feature_columns = [c for c in df.columns if c != target_column]
        self.is_fitted = True

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.

        Parameters
        ----------
        df : pd.DataFrame
            New data to transform.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()

        # Handle missing values
        df = handle_missing_values(
            df,
            numerical_strategy=self.numerical_impute_strategy,
            categorical_strategy=self.categorical_impute_strategy
        )

        # Encode categorical features
        df = encode_categorical_features(
            df,
            encoding_type=self.encoding_type
        )

        # Scale numerical features using fitted scaler
        if self.scale_features and self.scaler is not None:
            col_types = identify_column_types(df)
            df, _ = scale_numerical_features(
                df,
                numerical_columns=col_types['numerical'],
                scaler=self.scaler
            )

        return df

    def save(self, filepath: str) -> None:
        """Save the preprocessor to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ChurnPreprocessor':
        """Load a preprocessor from disk."""
        return joblib.load(filepath)

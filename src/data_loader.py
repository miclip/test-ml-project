"""
Data Loading Utilities for Customer Churn Prediction

This module provides functions to load and validate customer data
from CSV files for churn prediction modeling.

Expected Data Format:
---------------------
The input CSV should contain the following columns:
- customer_id: Unique identifier for each customer (string/int)
- tenure: Number of months the customer has been with the company (int)
- monthly_charges: Monthly billing amount (float)
- total_charges: Total amount billed (float)
- contract: Type of contract (categorical: 'Month-to-month', 'One year', 'Two year')
- payment_method: Payment method used (categorical)
- churn: Target variable - whether customer churned (0/1 or 'Yes'/'No')

Additional demographic and behavioral features may be included.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any


# Required columns that must be present in the dataset
REQUIRED_COLUMNS = ['customer_id', 'churn']

# Expected column types for validation
EXPECTED_TYPES = {
    'customer_id': ['object', 'int64', 'int32'],
    'tenure': ['int64', 'int32', 'float64'],
    'monthly_charges': ['float64', 'int64'],
    'total_charges': ['float64', 'object'],  # object allowed due to possible blanks
    'churn': ['object', 'int64', 'int32', 'bool'],
}


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


def load_data(
    filepath: str,
    validate: bool = True,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load customer data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing customer data.
    validate : bool, default=True
        Whether to run validation checks after loading.
    required_columns : list of str, optional
        List of columns that must be present. Defaults to REQUIRED_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Loaded customer data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    DataValidationError
        If validation is enabled and data fails validation checks.

    Examples
    --------
    >>> df = load_data('data/customers.csv')
    >>> print(df.shape)
    (7043, 21)
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load the CSV file
    df = pd.read_csv(filepath)

    if validate:
        if required_columns is None:
            required_columns = REQUIRED_COLUMNS
        validate_data(df, required_columns=required_columns)

    return df


def validate_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_types: bool = True
) -> Dict[str, Any]:
    """
    Validate the loaded dataset for required columns and data types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list of str, optional
        List of columns that must be present.
    check_types : bool, default=True
        Whether to check column data types.

    Returns
    -------
    dict
        Validation report containing checks performed and results.

    Raises
    ------
    DataValidationError
        If any required column is missing.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    validation_report = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_required_columns': [],
        'type_mismatches': [],
        'null_counts': {},
        'passed': True
    }

    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        validation_report['missing_required_columns'] = missing_cols
        validation_report['passed'] = False
        raise DataValidationError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check data types for known columns
    if check_types:
        for col, expected_types in EXPECTED_TYPES.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type not in expected_types:
                    validation_report['type_mismatches'].append({
                        'column': col,
                        'expected': expected_types,
                        'actual': actual_type
                    })

    # Count null values
    validation_report['null_counts'] = df.isnull().sum().to_dict()

    return validation_report


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to summarize.

    Returns
    -------
    dict
        Summary statistics including shape, dtypes, and null counts.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }

    # Add target variable distribution if present
    if 'churn' in df.columns:
        summary['target_distribution'] = df['churn'].value_counts().to_dict()

    return summary


def encode_target(
    df: pd.DataFrame,
    target_column: str = 'churn',
    positive_label: str = 'Yes'
) -> pd.DataFrame:
    """
    Encode the target variable to binary (0/1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column.
    target_column : str, default='churn'
        Name of the target column.
    positive_label : str, default='Yes'
        Label representing the positive class (churn).

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded target variable.
    """
    df = df.copy()

    if df[target_column].dtype == 'object':
        df[target_column] = (df[target_column] == positive_label).astype(int)
    elif df[target_column].dtype == 'bool':
        df[target_column] = df[target_column].astype(int)

    return df


def create_sample_data(
    n_samples: int = 1000,
    random_state: int = 42,
    churn_rate: float = 0.2
) -> pd.DataFrame:
    """
    Create sample customer data for testing and development.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate.
    random_state : int, default=42
        Random seed for reproducibility.
    churn_rate : float, default=0.2
        Proportion of churned customers.

    Returns
    -------
    pd.DataFrame
        Sample customer dataset.
    """
    np.random.seed(random_state)

    n_churned = int(n_samples * churn_rate)
    n_retained = n_samples - n_churned

    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 100, n_samples).round(2),
        'total_charges': np.random.uniform(100, 5000, n_samples).round(2),
        'contract': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            n_samples,
            p=[0.5, 0.3, 0.2]
        ),
        'payment_method': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
            n_samples
        ),
        'churn': [1] * n_churned + [0] * n_retained
    }

    df = pd.DataFrame(data)

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

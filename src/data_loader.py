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

    Generates realistic correlations:
    - Lower tenure -> higher churn probability
    - Month-to-month contracts -> higher churn probability
    - Higher monthly charges -> higher churn probability
    - Electronic check payment -> higher churn probability

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate.
    random_state : int, default=42
        Random seed for reproducibility.
    churn_rate : float, default=0.2
        Approximate proportion of churned customers.

    Returns
    -------
    pd.DataFrame
        Sample customer dataset with realistic feature-target correlations.
    """
    np.random.seed(random_state)

    # Generate base features
    customer_ids = [f'CUST_{i:05d}' for i in range(n_samples)]
    tenure = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.uniform(20, 100, n_samples).round(2)

    # Contract type (affects churn)
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples,
        p=[0.55, 0.25, 0.20]
    )

    # Payment method (affects churn)
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_samples,
        p=[0.35, 0.22, 0.22, 0.21]
    )

    # Calculate total charges based on tenure and monthly (with some noise)
    total_charges = (tenure * monthly_charges * np.random.uniform(0.8, 1.2, n_samples)).round(2)
    total_charges = np.clip(total_charges, 100, 8000)

    # Calculate churn probability based on features (realistic correlations)
    churn_prob = np.zeros(n_samples)

    # Lower tenure increases churn probability
    churn_prob += (72 - tenure) / 72 * 0.3

    # Higher monthly charges increase churn probability
    churn_prob += (monthly_charges - 20) / 80 * 0.2

    # Contract type affects churn
    contract_effect = np.where(
        contract == 'Month-to-month', 0.25,
        np.where(contract == 'One year', 0.1, 0.0)
    )
    churn_prob += contract_effect

    # Payment method affects churn
    payment_effect = np.where(payment_method == 'Electronic check', 0.15, 0.0)
    churn_prob += payment_effect

    # Add some randomness and normalize
    churn_prob += np.random.uniform(-0.1, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)

    # Adjust threshold to achieve target churn rate
    threshold = np.percentile(churn_prob, 100 * (1 - churn_rate))
    churn = (churn_prob >= threshold).astype(int)

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract,
        'payment_method': payment_method,
        'churn': churn
    })

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

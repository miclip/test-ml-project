"""Tests for data preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.preprocessing import (
    identify_column_types,
    handle_missing_values,
    encode_categorical_features,
    scale_numerical_features,
    create_train_val_test_split,
    prepare_features_and_target,
    ChurnPreprocessor,
    RANDOM_STATE
)
from src.data_loader import create_sample_data


class TestIdentifyColumnTypes:
    """Tests for identify_column_types function."""

    def test_identifies_numerical_columns(self):
        df = create_sample_data()
        col_types = identify_column_types(df)
        assert 'tenure' in col_types['numerical']
        assert 'monthly_charges' in col_types['numerical']

    def test_identifies_categorical_columns(self):
        df = create_sample_data()
        col_types = identify_column_types(df)
        assert 'contract' in col_types['categorical']
        assert 'payment_method' in col_types['categorical']

    def test_excludes_target_and_id(self):
        df = create_sample_data()
        col_types = identify_column_types(df)
        all_cols = col_types['numerical'] + col_types['categorical']
        assert 'churn' not in all_cols
        assert 'customer_id' not in all_cols


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""

    def test_handles_numerical_missing(self):
        df = create_sample_data()
        df.loc[0:5, 'tenure'] = np.nan
        result = handle_missing_values(df)
        assert result['tenure'].isna().sum() == 0

    def test_handles_categorical_missing(self):
        df = create_sample_data()
        df.loc[0:5, 'contract'] = np.nan
        result = handle_missing_values(df)
        assert result['contract'].isna().sum() == 0


class TestEncodeCategoricalFeatures:
    """Tests for encode_categorical_features function."""

    def test_onehot_encoding(self):
        df = create_sample_data(n_samples=100)
        result = encode_categorical_features(df, encoding_type='onehot')
        assert 'contract' not in result.columns
        assert any('contract_' in col for col in result.columns)

    def test_label_encoding(self):
        df = create_sample_data(n_samples=100)
        result = encode_categorical_features(df, encoding_type='label')
        assert 'contract' in result.columns
        assert result['contract'].dtype in ['int64', 'int32']


class TestScaleNumericalFeatures:
    """Tests for scale_numerical_features function."""

    def test_scales_to_standard(self):
        df = create_sample_data(n_samples=1000)
        result, scaler = scale_numerical_features(df)
        # Mean should be close to 0, std close to 1
        assert abs(result['tenure'].mean()) < 0.1
        assert abs(result['tenure'].std() - 1.0) < 0.1

    def test_returns_fitted_scaler(self):
        df = create_sample_data()
        _, scaler = scale_numerical_features(df)
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')


class TestCreateTrainValTestSplit:
    """Tests for create_train_val_test_split function."""

    def test_correct_split_ratios(self):
        df = create_sample_data(n_samples=1000)
        train, val, test = create_train_val_test_split(df)

        total = len(df)
        assert abs(len(train) / total - 0.70) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02

    def test_stratification_preserves_ratio(self):
        df = create_sample_data(n_samples=1000, churn_rate=0.2)
        train, val, test = create_train_val_test_split(df, stratify=True)

        original_rate = df['churn'].mean()
        train_rate = train['churn'].mean()
        val_rate = val['churn'].mean()
        test_rate = test['churn'].mean()

        assert abs(train_rate - original_rate) < 0.05
        assert abs(val_rate - original_rate) < 0.05
        assert abs(test_rate - original_rate) < 0.05

    def test_no_data_leakage(self):
        df = create_sample_data(n_samples=100)
        train, val, test = create_train_val_test_split(df)

        train_ids = set(train['customer_id'])
        val_ids = set(val['customer_id'])
        test_ids = set(test['customer_id'])

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestPrepareeFeaturesAndTarget:
    """Tests for prepare_features_and_target function."""

    def test_separates_correctly(self):
        df = create_sample_data()
        X, y = prepare_features_and_target(df)
        assert 'churn' not in X.columns
        assert 'customer_id' not in X.columns
        assert len(y) == len(df)


class TestChurnPreprocessor:
    """Tests for ChurnPreprocessor class."""

    def test_fit_transform(self):
        df = create_sample_data()
        preprocessor = ChurnPreprocessor()
        result = preprocessor.fit_transform(df)
        assert preprocessor.is_fitted
        assert len(result) == len(df)

    def test_transform_after_fit(self):
        df = create_sample_data(n_samples=100)
        preprocessor = ChurnPreprocessor()
        preprocessor.fit_transform(df)

        new_df = create_sample_data(n_samples=50, random_state=99)
        result = preprocessor.transform(new_df)
        assert len(result) == 50

    def test_transform_without_fit_raises(self):
        preprocessor = ChurnPreprocessor()
        df = create_sample_data()
        with pytest.raises(ValueError):
            preprocessor.transform(df)

    def test_save_and_load(self):
        df = create_sample_data()
        preprocessor = ChurnPreprocessor()
        preprocessor.fit_transform(df)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            preprocessor.save(f.name)
            loaded = ChurnPreprocessor.load(f.name)
            assert loaded.is_fitted
            Path(f.name).unlink()

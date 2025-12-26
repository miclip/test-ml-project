"""Tests for data loading utilities."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data_loader import (
    load_data,
    validate_data,
    get_data_summary,
    encode_target,
    create_sample_data,
    DataValidationError,
    REQUIRED_COLUMNS
)


class TestCreateSampleData:
    """Tests for create_sample_data function."""

    def test_creates_correct_number_of_samples(self):
        df = create_sample_data(n_samples=500)
        assert len(df) == 500

    def test_creates_required_columns(self):
        df = create_sample_data()
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_churn_rate_approximate(self):
        df = create_sample_data(n_samples=1000, churn_rate=0.3)
        actual_rate = df['churn'].mean()
        assert 0.25 <= actual_rate <= 0.35  # Allow some variance

    def test_reproducibility_with_random_state(self):
        df1 = create_sample_data(random_state=42)
        df2 = create_sample_data(random_state=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestValidateData:
    """Tests for validate_data function."""

    def test_valid_data_passes(self):
        df = create_sample_data()
        report = validate_data(df)
        assert report['passed'] is True

    def test_missing_column_raises_error(self):
        df = create_sample_data()
        df = df.drop(columns=['customer_id'])
        with pytest.raises(DataValidationError):
            validate_data(df)

    def test_reports_null_counts(self):
        df = create_sample_data()
        df.loc[0:10, 'tenure'] = np.nan
        report = validate_data(df, required_columns=['customer_id', 'churn'])
        assert report['null_counts']['tenure'] == 11


class TestLoadData:
    """Tests for load_data function."""

    def test_file_not_found_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.csv')

    def test_loads_valid_csv(self):
        df = create_sample_data(n_samples=100)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            loaded_df = load_data(f.name)
            assert len(loaded_df) == 100
            Path(f.name).unlink()


class TestEncodeTarget:
    """Tests for encode_target function."""

    def test_encodes_string_target(self):
        df = pd.DataFrame({'churn': ['Yes', 'No', 'Yes', 'No']})
        encoded = encode_target(df)
        assert list(encoded['churn']) == [1, 0, 1, 0]

    def test_preserves_numeric_target(self):
        df = pd.DataFrame({'churn': [1, 0, 1, 0]})
        encoded = encode_target(df)
        assert list(encoded['churn']) == [1, 0, 1, 0]


class TestGetDataSummary:
    """Tests for get_data_summary function."""

    def test_returns_shape(self):
        df = create_sample_data(n_samples=100)
        summary = get_data_summary(df)
        assert summary['shape'] == (100, 7)

    def test_includes_target_distribution(self):
        df = create_sample_data()
        summary = get_data_summary(df)
        assert 'target_distribution' in summary

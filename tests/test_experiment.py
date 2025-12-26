"""Tests for MLflow experiment tracking utilities."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.experiment import (
    setup_experiment,
    start_run,
    log_params,
    log_metrics,
    log_classification_metrics,
    log_dataset_info,
    get_best_run,
    DEFAULT_EXPERIMENT,
    CLASSIFICATION_METRICS
)


class TestSetupExperiment:
    """Tests for setup_experiment function."""

    def test_creates_experiment(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            exp_id = setup_experiment("test_experiment")
            assert exp_id is not None

    def test_returns_existing_experiment(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            exp_id1 = setup_experiment("test_experiment_2")
            exp_id2 = setup_experiment("test_experiment_2")
            assert exp_id1 == exp_id2


class TestStartRun:
    """Tests for start_run function."""

    def test_creates_run_with_name(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            setup_experiment("test_run_exp")
            with start_run(run_name="test_run") as run:
                assert run.info.run_name == "test_run"

    def test_auto_generates_run_name(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            setup_experiment("test_run_exp_2")
            with start_run() as run:
                assert run.info.run_name.startswith("run_")


class TestLogParams:
    """Tests for log_params function."""

    def test_logs_dict_params(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            setup_experiment("test_params")
            with start_run():
                log_params({"param1": "value1", "param2": 42})
                # No assertion needed - just verify no errors


class TestLogMetrics:
    """Tests for log_metrics function."""

    def test_logs_multiple_metrics(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            setup_experiment("test_metrics")
            with start_run():
                log_metrics({"accuracy": 0.85, "f1": 0.82})


class TestLogClassificationMetrics:
    """Tests for log_classification_metrics function."""

    def test_computes_all_metrics(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            setup_experiment("test_classification")
            with start_run():
                y_true = np.array([0, 0, 1, 1, 1])
                y_pred = np.array([0, 1, 1, 1, 0])
                y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.4])

                metrics = log_classification_metrics(y_true, y_pred, y_prob)

                assert 'accuracy' in metrics
                assert 'precision' in metrics
                assert 'recall' in metrics
                assert 'f1_score' in metrics
                assert 'roc_auc' in metrics

    def test_handles_prefix(self):
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            setup_experiment("test_prefix")
            with start_run():
                y_true = np.array([0, 1, 1])
                y_pred = np.array([0, 1, 0])

                metrics = log_classification_metrics(
                    y_true, y_pred, prefix="val_"
                )

                assert 'val_accuracy' in metrics


class TestLogDatasetInfo:
    """Tests for log_dataset_info function."""

    def test_logs_dataset_params(self):
        import mlflow
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            setup_experiment("test_dataset")
            with start_run():
                df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
                log_dataset_info(df, name="test_data")


class TestConstants:
    """Tests for module constants."""

    def test_default_experiment_name(self):
        assert DEFAULT_EXPERIMENT == "churn_prediction"

    def test_classification_metrics_defined(self):
        assert 'accuracy' in CLASSIFICATION_METRICS
        assert 'roc_auc' in CLASSIFICATION_METRICS

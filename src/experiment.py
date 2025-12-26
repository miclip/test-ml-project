"""
MLflow Experiment Tracking Utilities for Customer Churn Prediction

This module provides functions for experiment tracking, logging metrics,
and managing model artifacts with MLflow.

Naming Conventions:
- Experiment: churn_prediction_{model_type}_{YYYYMMDD}
- Run: {model_name}_{timestamp}
- Artifacts: model_{algorithm}_{version}
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path


# Default experiment name
DEFAULT_EXPERIMENT = "churn_prediction"

# Standard metrics to log for classification
CLASSIFICATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
]


def setup_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None
) -> str:
    """
    Set up an MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    tracking_uri : str, optional
        URI for the MLflow tracking server. Defaults to local ./mlruns.
    artifact_location : str, optional
        Location for storing artifacts.

    Returns
    -------
    str
        The experiment ID.

    Examples
    --------
    >>> experiment_id = setup_experiment("churn_prediction_baseline")
    >>> print(f"Experiment ID: {experiment_id}")
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Create or get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return experiment_id


def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    description: Optional[str] = None
) -> mlflow.ActiveRun:
    """
    Start an MLflow run with optional metadata.

    Parameters
    ----------
    run_name : str, optional
        Name for the run. Auto-generated if not provided.
    tags : dict, optional
        Tags to attach to the run.
    description : str, optional
        Description of the run.

    Returns
    -------
    mlflow.ActiveRun
        The active run context.

    Examples
    --------
    >>> with start_run(run_name="logistic_regression_v1") as run:
    ...     # Log parameters and metrics
    ...     pass
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_tags = tags or {}
    if description:
        run_tags["mlflow.note.content"] = description

    return mlflow.start_run(run_name=run_name, tags=run_tags)


def log_params(params: Dict[str, Any]) -> None:
    """
    Log multiple parameters to the current run.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values.
    """
    for key, value in params.items():
        # Convert non-string values to strings
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        mlflow.log_param(key, value)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log multiple metrics to the current run.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values.
    step : int, optional
        Step number for the metrics.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def log_classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate and log standard classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.
    prefix : str
        Prefix for metric names (e.g., 'val_' for validation metrics).

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )

    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_prob)

    log_metrics(metrics)

    return metrics


def log_model(
    model,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
    input_example=None
) -> None:
    """
    Log a scikit-learn model to MLflow.

    Parameters
    ----------
    model : sklearn estimator
        The trained model to log.
    artifact_path : str
        Path within the run's artifact directory.
    registered_model_name : str, optional
        If provided, register the model with this name.
    input_example : array-like, optional
        Example input for the model signature.
    """
    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        input_example=input_example
    )


def log_figure(fig, filename: str) -> None:
    """
    Log a matplotlib figure as an artifact.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to log.
    filename : str
        Filename for the artifact (e.g., 'confusion_matrix.png').
    """
    mlflow.log_figure(fig, filename)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a local file as an artifact.

    Parameters
    ----------
    local_path : str
        Path to the local file.
    artifact_path : str, optional
        Directory within artifacts to place the file.
    """
    mlflow.log_artifact(local_path, artifact_path)


def log_dataset_info(
    df,
    name: str = "training_data",
    description: Optional[str] = None
) -> None:
    """
    Log dataset information as parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    name : str
        Name identifier for the dataset.
    description : str, optional
        Description of the dataset.
    """
    import hashlib

    params = {
        f"{name}_rows": len(df),
        f"{name}_columns": len(df.columns),
        f"{name}_hash": hashlib.md5(
            str(df.values.tobytes()).encode()
        ).hexdigest()[:8]
    }

    if description:
        params[f"{name}_description"] = description

    log_params(params)


def get_best_run(
    experiment_name: str,
    metric: str = "roc_auc",
    ascending: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Get the best run from an experiment based on a metric.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    metric : str
        Metric to optimize.
    ascending : bool
        If True, lower is better. If False, higher is better.

    Returns
    -------
    dict or None
        Best run info or None if no runs found.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )

    if not runs:
        return None

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "run_name": best_run.info.run_name,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
    }


def load_model(run_id: str, artifact_path: str = "model"):
    """
    Load a model from a specific run.

    Parameters
    ----------
    run_id : str
        The run ID containing the model.
    artifact_path : str
        Path to the model artifact within the run.

    Returns
    -------
    sklearn estimator
        The loaded model.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(model_uri)

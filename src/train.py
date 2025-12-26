"""
Model training script for churn prediction pipeline.

This script provides a command-line interface for training churn prediction
models with configurable parameters and MLflow experiment tracking.

Usage:
    python -m src.train --model random_forest --n-estimators 100
    python -m src.train --model gradient_boosting --learning-rate 0.1
    python -m src.train --model logistic_regression
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data_loader import create_sample_data, load_data
from src.preprocessing import (
    ChurnPreprocessor,
    create_train_val_test_split,
    prepare_features_and_target
)
from src.experiment import (
    setup_experiment,
    start_run,
    log_params,
    log_classification_metrics,
    log_model,
    log_dataset_info
)


# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'class': LogisticRegression,
        'default_params': {
            'max_iter': 1000,
            'random_state': 42
        }
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'default_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'default_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'subsample': 0.8,
            'random_state': 42
        }
    }
}


def get_model(model_type: str, **kwargs) -> Any:
    """
    Get a model instance with specified parameters.

    Args:
        model_type: Type of model ('logistic_regression', 'random_forest', 'gradient_boosting')
        **kwargs: Model-specific parameters to override defaults

    Returns:
        Configured model instance

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_type]
    params = {**config['default_params'], **kwargs}
    return config['class'](**params)


def prepare_data(
    data_path: Optional[str] = None,
    n_samples: int = 1000,
    random_state: int = 42,
    churn_rate: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare data for training.

    Args:
        data_path: Path to CSV data file. If None, generates sample data.
        n_samples: Number of samples for generated data
        random_state: Random seed for reproducibility
        churn_rate: Churn rate for generated data

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    # Load or generate data
    if data_path:
        df = load_data(data_path)
    else:
        df = create_sample_data(
            n_samples=n_samples,
            random_state=random_state,
            churn_rate=churn_rate
        )

    # Split data
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        target_column='churn',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=random_state
    )

    # Preprocess
    preprocessor = ChurnPreprocessor()

    train_preprocessed = preprocessor.fit_transform(train_df, target_column='churn')
    X_train, y_train = prepare_features_and_target(train_preprocessed, target_column='churn')

    val_preprocessed = preprocessor.transform(val_df)
    X_val, y_val = prepare_features_and_target(val_preprocessed, target_column='churn')

    test_preprocessed = preprocessor.transform(test_df)
    X_test, y_test = prepare_features_and_target(test_preprocessed, target_column='churn')

    feature_names = list(X_train.columns)

    return (
        X_train.values, X_val.values, X_test.values,
        y_train.values, y_val.values, y_test.values,
        feature_names
    )


def train_model(
    model_type: str,
    experiment_name: str = "churn_prediction",
    run_name: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = "models",
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a churn prediction model with MLflow tracking.

    Args:
        model_type: Type of model to train
        experiment_name: MLflow experiment name
        run_name: Optional run name (auto-generated if not provided)
        data_path: Path to training data (generates sample if None)
        output_dir: Directory to save model artifacts
        model_params: Model-specific parameters
        random_state: Random seed

    Returns:
        Dictionary with training results including metrics and run info
    """
    model_params = model_params or {}

    # Prepare data
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(
        data_path=data_path,
        random_state=random_state
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Get model
    model = get_model(model_type, **model_params)

    # Setup MLflow
    setup_experiment(experiment_name)

    run_name = run_name or f"train_{model_type}"

    with start_run(run_name=run_name) as run:
        # Log parameters
        all_params = {
            'model_type': model_type,
            'random_state': random_state,
            **MODEL_CONFIGS[model_type]['default_params'],
            **model_params
        }
        log_params(all_params)

        # Train model
        print(f"Training {model_type}...")
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        val_metrics = log_classification_metrics(
            y_val, y_val_pred, y_val_prob, prefix="val_"
        )

        print(f"Validation AUC-ROC: {val_metrics['val_roc_auc']:.4f}")

        # Log model to MLflow
        log_model(model, artifact_path="model")

        run_id = run.info.run_id

    # Save model locally
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / f"{model_type}_model.joblib"
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")

    # Save metadata
    metadata = {
        'model_type': model_type,
        'run_id': run_id,
        'feature_names': feature_names,
        'val_metrics': val_metrics,
        'params': all_params
    }

    metadata_file = output_path / f"{model_type}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'model': model,
        'run_id': run_id,
        'val_metrics': val_metrics,
        'model_file': str(model_file),
        'metadata_file': str(metadata_file)
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a churn prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['logistic_regression', 'random_forest', 'gradient_boosting'],
        default='random_forest',
        help='Model type to train'
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='churn_prediction',
        help='MLflow experiment name'
    )

    parser.add_argument(
        '--run-name', '-r',
        type=str,
        default=None,
        help='MLflow run name (auto-generated if not specified)'
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='Path to CSV data file (generates sample data if not specified)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models',
        help='Directory to save model artifacts'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Model-specific parameters
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=None,
        help='Number of estimators (for RF/GB)'
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Maximum tree depth'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (for GB)'
    )

    parser.add_argument(
        '--min-samples-split',
        type=int,
        default=None,
        help='Minimum samples to split'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build model params from CLI args
    model_params = {}
    if args.n_estimators is not None:
        model_params['n_estimators'] = args.n_estimators
    if args.max_depth is not None:
        model_params['max_depth'] = args.max_depth
    if args.learning_rate is not None:
        model_params['learning_rate'] = args.learning_rate
    if args.min_samples_split is not None:
        model_params['min_samples_split'] = args.min_samples_split

    print(f"Training {args.model} model...")
    print(f"Experiment: {args.experiment}")

    try:
        results = train_model(
            model_type=args.model,
            experiment_name=args.experiment,
            run_name=args.run_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_params=model_params,
            random_state=args.random_state
        )

        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Run ID: {results['run_id']}")
        print(f"Val AUC-ROC: {results['val_metrics']['val_roc_auc']:.4f}")
        print(f"Model saved: {results['model_file']}")
        print(f"Metadata saved: {results['metadata_file']}")

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

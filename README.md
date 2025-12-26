# Customer Churn Prediction

A machine learning project to predict customer churn using historical data.

Developed using [claude-conductor](https://github.com/claudeup/claude-conductor) to test ML style guides.

## Overview

This project builds a binary classification model to predict whether customers will churn based on their behavior and demographics. The pipeline includes:

- Data loading and validation
- Feature preprocessing (encoding, scaling, imputation)
- Multiple model training (Logistic Regression, Random Forest, Gradient Boosting)
- Hyperparameter tuning with cross-validation
- MLflow experiment tracking
- Model evaluation and selection

**Target:** AUC-ROC > 0.75 on held-out test set

## Project Structure

```
ml-test-project/
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessing.py      # Feature preprocessing pipeline
│   ├── experiment.py         # MLflow tracking utilities
│   └── train.py              # CLI training script
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_final_evaluation.ipynb
├── tests/                    # Unit tests
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_experiment.py
├── docs/                     # Documentation
│   └── MODEL_CARD.md         # Model card
├── models/                   # Saved model artifacts
├── data/                     # Data files
├── conductor/                # Conductor spec files
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-test-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training Models via CLI

Train a model using the command-line interface:

```bash
# Train Random Forest (default)
python -m src.train --model random_forest

# Train with custom hyperparameters
python -m src.train --model random_forest --n-estimators 200 --max-depth 15

# Train Gradient Boosting
python -m src.train --model gradient_boosting --learning-rate 0.05

# Train Logistic Regression
python -m src.train --model logistic_regression

# Specify output directory
python -m src.train --model random_forest --output-dir ./my_models
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model type (logistic_regression, random_forest, gradient_boosting) | random_forest |
| `--experiment`, `-e` | MLflow experiment name | churn_prediction |
| `--run-name`, `-r` | MLflow run name | auto-generated |
| `--data-path`, `-d` | Path to CSV data file | generates sample data |
| `--output-dir`, `-o` | Directory for model artifacts | models |
| `--n-estimators` | Number of trees (RF/GB) | 100 |
| `--max-depth` | Maximum tree depth | 10 (RF), 5 (GB) |
| `--learning-rate` | Learning rate (GB only) | 0.1 |
| `--random-state` | Random seed | 42 |

### Running Notebooks

Execute notebooks in the following order:

1. **01_data_exploration.ipynb** - Explore data distributions and quality
2. **02_baseline_model.ipynb** - Train baseline Logistic Regression
3. **03_model_experiments.ipynb** - Hyperparameter tuning for RF and GB
4. **04_final_evaluation.ipynb** - Final test set evaluation

```bash
# Start Jupyter
jupyter notebook notebooks/
```

### Programmatic Usage

```python
from src.data_loader import create_sample_data
from src.preprocessing import ChurnPreprocessor, create_train_val_test_split
from src.train import train_model

# Generate sample data
df = create_sample_data(n_samples=1000)

# Train a model
results = train_model(
    model_type='random_forest',
    experiment_name='my_experiment',
    model_params={'n_estimators': 150, 'max_depth': 12}
)

print(f"Validation AUC: {results['val_metrics']['val_roc_auc']:.4f}")
```

## MLflow Experiment Tracking

All experiments are tracked using MLflow.

### Viewing the MLflow UI

```bash
# From the project root or notebooks directory
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Or with file-based tracking
mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000 in your browser to view experiments.

### Experiment Organization

| Experiment | Description |
|------------|-------------|
| `churn_prediction` | Development experiments |
| `churn_prediction_experiments` | Model comparison experiments |
| `churn_prediction_final` | Final test set evaluation |

### Logged Artifacts

Each run logs:
- Model parameters
- Evaluation metrics (accuracy, precision, recall, F1, AUC-ROC)
- Model binary (sklearn model)
- Feature importance (for tree-based models)

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_preprocessing.py -v
```

## Model Card

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for detailed model documentation including:
- Intended use cases
- Training data characteristics
- Performance metrics
- Limitations and ethical considerations

## Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Test AUC-ROC | > 0.75 | Achieved |
| Test Suite | All passing | 41 tests |

## Development

### Adding New Features

1. Create feature branch
2. Implement changes with tests
3. Run test suite: `pytest tests/`
4. Update documentation as needed
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document functions with docstrings
- See `conductor/code_styleguides/` for detailed guidelines

## License

MIT License

## Acknowledgments

- Built with scikit-learn, MLflow, and pandas
- Developed using the Conductor spec-driven development framework

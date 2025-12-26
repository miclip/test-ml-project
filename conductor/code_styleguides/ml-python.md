# ML Python Style Guide Summary

This document extends the Python style guide with machine learning and data science conventions.

## 1. General Principles
- **Reproducibility:** Set random seeds, version data and models, pin dependencies.
- **Clarity:** Prefer readable code over clever optimizations.
- **Documentation:** Explain why, not just what. Document experiment decisions.

## 2. Project Structure
- `data/` - raw, processed, and feature data (raw is immutable)
- `notebooks/` - exploration and reports (numbered: `01_eda.ipynb`)
- `src/` - production code (data, features, models, training, evaluation)
- `models/` - trained model artifacts
- `configs/` - configuration files

## 3. Jupyter Notebooks
- **Structure:** Start with purpose/overview, then setup, data loading, analysis, conclusions.
- **First Cell:** Imports, random seeds, display settings, `%matplotlib inline`.
- **Cells:** One concept per cell. Keep under 20 lines. Use markdown headers.
- **Outputs:** Clear outputs before committing exploration notebooks. Keep outputs for reports.
- **Version Control:** Use `nbstripout` or `jupytext` to avoid noisy diffs.
- **Graduation:** Move reusable code (>20 lines) to `src/` modules.

## 4. Data Handling
- **Loading:** Use explicit functions with documented paths and versions.
- **Validation:** Validate data at boundaries (schema, nulls, ranges).
- **No Leakage:** Fit preprocessors on training data only. Never peek at test data.

## 5. Experiment Tracking
- Log all experiments to MLflow, W&B, or similar.
- Record: hyperparameters, metrics, data version, model artifacts.
- Use config files (YAML) for experiment parameters.

## 6. Model Code
- Use configuration objects or dataclasses for model parameters.
- Separate training, evaluation, and inference functions.
- Document model inputs, outputs, and assumptions.

## 7. Testing
- Test data preprocessing and feature engineering functions.
- Test model inference with known inputs/outputs.
- Use `pytest --nbval` to verify notebooks run without errors.

## 8. Naming
- Features: `num_`, `cat_`, `bool_` prefixes (e.g., `num_transactions_30d`).
- Models: Include version and date (e.g., `churn_model_v2_20240115.pkl`).
- Experiments: Descriptive names in tracking system.

**BE CONSISTENT.** Match existing project conventions.

*References: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)*

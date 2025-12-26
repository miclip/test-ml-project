# Plan: Build the Core Churn Prediction Pipeline

## Track Overview
**Track ID:** churn_pipeline_20251226
**Description:** Implement data loading, preprocessing, model training, and evaluation with MLflow experiment tracking
**Target:** AUC-ROC > 0.75 on test set

---

## Phase 1: Project Setup and Data Pipeline

### Objective
Set up the project structure, configure MLflow, and create a robust data loading and preprocessing pipeline.

### Tasks

- [x] Task 1.1: Set up project structure (ad30d17)
  - Create directory structure: `src/`, `notebooks/`, `data/`, `models/`, `tests/`
  - Initialize `src/__init__.py` and module files
  - Verify MLflow is configured correctly

- [ ] Task 1.2: Implement data loading utilities
  - Create `src/data_loader.py` with CSV loading functions
  - Add data validation checks (schema, types, required columns)
  - Document expected data format

- [ ] Task 1.3: Implement data preprocessing pipeline
  - Create `src/preprocessing.py` with preprocessing functions
  - Handle missing values (imputation strategy)
  - Encode categorical variables (one-hot or label encoding)
  - Scale numerical features (StandardScaler)
  - Create train/validation/test split with stratification (70/15/15)

- [ ] Task 1.4: Create data exploration notebook
  - Create `notebooks/01_data_exploration.ipynb`
  - Analyze feature distributions
  - Check class balance of target variable
  - Identify potential data quality issues
  - Generate correlation heatmap

- [ ] Task: Conductor - User Manual Verification 'Phase 1: Project Setup and Data Pipeline' (Protocol in workflow.md)

---

## Phase 2: Baseline Model Development

### Objective
Establish a baseline model and set up experiment tracking with MLflow.

### Tasks

- [ ] Task 2.1: Configure MLflow experiment tracking
  - Create `src/experiment.py` with MLflow utilities
  - Set up experiment naming conventions
  - Define standard metrics to log

- [ ] Task 2.2: Train baseline Logistic Regression model
  - Create `notebooks/02_baseline_model.ipynb`
  - Define success criteria (baseline to beat)
  - Train Logistic Regression with default parameters
  - Log hyperparameters and metrics to MLflow
  - Evaluate on validation set
  - Generate confusion matrix and ROC curve

- [ ] Task 2.3: Analyze baseline results
  - Document baseline performance metrics
  - Analyze misclassifications
  - Identify potential improvements
  - Record experiment ID in notebook

- [ ] Task: Conductor - User Manual Verification 'Phase 2: Baseline Model Development' (Protocol in workflow.md)

---

## Phase 3: Model Improvement and Evaluation

### Objective
Improve upon baseline with tree-based models and perform thorough evaluation.

### Tasks

- [ ] Task 3.1: Train Random Forest classifier
  - Create `notebooks/03_model_experiments.ipynb`
  - Train Random Forest with hyperparameter tuning
  - Use cross-validation for robust evaluation
  - Log all experiments to MLflow
  - Compare against baseline

- [ ] Task 3.2: Train Gradient Boosting classifier
  - Train GradientBoostingClassifier or XGBoost
  - Tune key hyperparameters (n_estimators, learning_rate, max_depth)
  - Log experiments to MLflow
  - Compare against baseline and Random Forest

- [ ] Task 3.3: Select best model and evaluate on test set
  - Compare all model performances on validation set
  - Select best performing model
  - Evaluate ONCE on held-out test set
  - Generate final metrics: accuracy, precision, recall, F1, AUC-ROC
  - Verify AUC-ROC > 0.75 threshold met

- [ ] Task 3.4: Generate feature importance analysis
  - Extract feature importances from best model
  - Create feature importance visualization
  - Document top contributing features
  - Validate alignment with domain knowledge

- [ ] Task: Conductor - User Manual Verification 'Phase 3: Model Improvement and Evaluation' (Protocol in workflow.md)

---

## Phase 4: Documentation and Finalization

### Objective
Complete documentation and ensure reproducibility.

### Tasks

- [ ] Task 4.1: Create model training script
  - Create `src/train.py` with training pipeline
  - Accept configuration parameters
  - Integrate MLflow logging
  - Save model artifacts

- [ ] Task 4.2: Create model card
  - Document model description and intended use
  - Record training data characteristics
  - List performance metrics across data slices
  - Document limitations and ethical considerations
  - Include version and lineage information

- [ ] Task 4.3: Update project README
  - Document project setup instructions
  - Add usage examples
  - Include MLflow UI instructions
  - Document notebook execution order

- [ ] Task 4.4: Final validation and cleanup
  - Verify all notebooks run top-to-bottom
  - Ensure all experiments are logged to MLflow
  - Clean up any temporary files
  - Verify reproducibility with fresh environment

- [ ] Task: Conductor - User Manual Verification 'Phase 4: Documentation and Finalization' (Protocol in workflow.md)

---

## Quality Gates

Before marking this track complete:
- [ ] Data pipeline processes data without errors
- [ ] Model achieves AUC-ROC > 0.75 on test set
- [ ] All experiments logged to MLflow with metrics and parameters
- [ ] Feature importance visualization generated
- [ ] All notebooks runnable top-to-bottom
- [ ] Model card created
- [ ] README updated with usage instructions

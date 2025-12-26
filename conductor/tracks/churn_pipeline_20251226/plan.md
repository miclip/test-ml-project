# Plan: Build the Core Churn Prediction Pipeline

## Track Overview
**Track ID:** churn_pipeline_20251226
**Description:** Implement data loading, preprocessing, model training, and evaluation with MLflow experiment tracking
**Target:** AUC-ROC > 0.75 on test set

---

## Phase 1: Project Setup and Data Pipeline [checkpoint: 26fa336]

### Objective
Set up the project structure, configure MLflow, and create a robust data loading and preprocessing pipeline.

### Tasks

- [x] Task 1.1: Set up project structure (ad30d17)
  - Create directory structure: `src/`, `notebooks/`, `data/`, `models/`, `tests/`
  - Initialize `src/__init__.py` and module files
  - Verify MLflow is configured correctly

- [x] Task 1.2: Implement data loading utilities (fd46e36)
  - Create `src/data_loader.py` with CSV loading functions
  - Add data validation checks (schema, types, required columns)
  - Document expected data format

- [x] Task 1.3: Implement data preprocessing pipeline (a89e40d)
  - Create `src/preprocessing.py` with preprocessing functions
  - Handle missing values (imputation strategy)
  - Encode categorical variables (one-hot or label encoding)
  - Scale numerical features (StandardScaler)
  - Create train/validation/test split with stratification (70/15/15)

- [x] Task 1.4: Create data exploration notebook (3de2b1d)
  - Create `notebooks/01_data_exploration.ipynb`
  - Analyze feature distributions
  - Check class balance of target variable
  - Identify potential data quality issues
  - Generate correlation heatmap

- [x] Task: Conductor - User Manual Verification 'Phase 1: Project Setup and Data Pipeline' (Protocol in workflow.md)

---

## Phase 2: Baseline Model Development [checkpoint: a6eac05]

### Objective
Establish a baseline model and set up experiment tracking with MLflow.

### Tasks

- [x] Task 2.1: Configure MLflow experiment tracking (67b35a4)
  - Create `src/experiment.py` with MLflow utilities
  - Set up experiment naming conventions
  - Define standard metrics to log

- [x] Task 2.2: Train baseline Logistic Regression model (ae2971f)
  - Create `notebooks/02_baseline_model.ipynb`
  - Define success criteria (baseline to beat)
  - Train Logistic Regression with default parameters
  - Log hyperparameters and metrics to MLflow
  - Evaluate on validation set
  - Generate confusion matrix and ROC curve

- [x] Task 2.3: Analyze baseline results (85e52b1)
  - Document baseline performance metrics
  - Analyze misclassifications
  - Identify potential improvements
  - Record experiment ID in notebook

- [x] Task: Conductor - User Manual Verification 'Phase 2: Baseline Model Development' (Protocol in workflow.md)

---

## Phase 3: Model Improvement and Evaluation [checkpoint: 75641e2]

### Objective
Improve upon baseline with tree-based models and perform thorough evaluation.

### Tasks

- [x] Task 3.1: Train Random Forest classifier (14cb448)
  - Create `notebooks/03_model_experiments.ipynb`
  - Train Random Forest with hyperparameter tuning
  - Use cross-validation for robust evaluation
  - Log all experiments to MLflow
  - Compare against baseline

- [x] Task 3.2: Train Gradient Boosting classifier (14cb448)
  - Train GradientBoostingClassifier or XGBoost
  - Tune key hyperparameters (n_estimators, learning_rate, max_depth)
  - Log experiments to MLflow
  - Compare against baseline and Random Forest

- [x] Task 3.3: Select best model and evaluate on test set (3967d30)
  - Compare all model performances on validation set
  - Select best performing model
  - Evaluate ONCE on held-out test set
  - Generate final metrics: accuracy, precision, recall, F1, AUC-ROC
  - Verify AUC-ROC > 0.75 threshold met

- [x] Task 3.4: Generate feature importance analysis (1c518c5)
  - Extract feature importances from best model
  - Create feature importance visualization
  - Document top contributing features
  - Validate alignment with domain knowledge

- [x] Task: Conductor - User Manual Verification 'Phase 3: Model Improvement and Evaluation' (Protocol in workflow.md)

---

## Phase 4: Documentation and Finalization

### Objective
Complete documentation and ensure reproducibility.

### Tasks

- [x] Task 4.1: Create model training script (34240de)
  - Create `src/train.py` with training pipeline
  - Accept configuration parameters
  - Integrate MLflow logging
  - Save model artifacts

- [x] Task 4.2: Create model card (b93cdb1)
  - Document model description and intended use
  - Record training data characteristics
  - List performance metrics across data slices
  - Document limitations and ethical considerations
  - Include version and lineage information

- [x] Task 4.3: Update project README (91b2a3b)
  - Document project setup instructions
  - Add usage examples
  - Include MLflow UI instructions
  - Document notebook execution order

- [x] Task 4.4: Final validation and cleanup (ccdbaef)
  - Verify all notebooks run top-to-bottom
  - Ensure all experiments are logged to MLflow
  - Clean up any temporary files
  - Verify reproducibility with fresh environment

- [~] Task: Conductor - User Manual Verification 'Phase 4: Documentation and Finalization' (Protocol in workflow.md)

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

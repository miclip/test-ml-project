# Product Guidelines: Customer Churn Prediction

## Communication Style

### Tone: Technical/Precise
- Use statistical terminology and detailed metrics when presenting results
- Include confidence intervals, p-values, and model coefficients where relevant
- Example: "Customer has 0.87 churn probability (95% CI: 0.82-0.91) based on logistic regression with AUC-ROC of 0.84"
- Provide precise numerical values rather than qualitative descriptions
- Reference specific features and their statistical significance

### Terminology Standards
- Use standard ML terminology: precision, recall, F1-score, AUC-ROC
- Reference features by their exact column names from the dataset
- Document statistical assumptions and methodology in detail

## Visual Identity

### Data Visualization Priority
- **Primary Output Format:** Charts and plots for model interpretation
- **Required Visualizations:**
  - Feature importance plots (bar charts or SHAP summary plots)
  - ROC curves with AUC scores
  - Confusion matrices with precision/recall metrics
  - Churn probability distributions (histograms)
  - Correlation heatmaps for feature analysis
- **Color Palette:** Use matplotlib defaults with consistent styling
- **Annotations:** Include axis labels, titles, and legends on all plots

## Documentation Structure

### Research-Style Approach
- **Primary Artifacts:** Jupyter notebooks with inline documentation
- **Notebook Structure:**
  1. Introduction and objective (markdown)
  2. Data loading and exploration
  3. Methodology explanation (markdown cells)
  4. Implementation with inline comments
  5. Results and interpretation (markdown)
  6. Conclusions and next steps
- **Code Comments:** Explain the "why" not just the "what"
- **Markdown Cells:** Use for methodology explanations, assumptions, and interpretations

## Naming Conventions

### Descriptive with Timestamps
- **Models:** `churn_model_{algorithm}_{YYYYMMDD}` (e.g., `churn_model_rf_20231215`)
- **Experiments:** `experiment_{description}_{version}` (e.g., `experiment_feature_selection_v2`)
- **Data Files:** `{dataset}_{stage}_{YYYYMMDD}` (e.g., `customers_cleaned_20231210`)
- **Notebooks:** `{number}_{description}.ipynb` (e.g., `01_data_exploration.ipynb`)
- **MLflow Runs:** Use descriptive run names matching the experiment naming pattern

## Quality Standards

- All visualizations must have proper labels and titles
- Notebooks must be runnable from top to bottom without errors
- Document data sources and preprocessing steps
- Include model performance metrics in experiment tracking

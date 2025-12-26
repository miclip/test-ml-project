# Model Card: Customer Churn Prediction

## Model Details

### Overview
- **Model Name:** Customer Churn Prediction Model
- **Version:** 1.0.0
- **Type:** Binary Classification
- **Framework:** scikit-learn
- **License:** MIT

### Model Architecture
The production model is selected from three candidates based on validation AUC-ROC:
- Logistic Regression (baseline)
- Random Forest Classifier
- Gradient Boosting Classifier

Best performing model is selected automatically based on validation set performance.

### Developers
- Development Team
- Contact: [team@example.com]

### Model Date
- Training Date: December 2025
- Last Updated: December 2025

---

## Intended Use

### Primary Use Cases
- **Customer Retention:** Identify customers at high risk of churning to enable proactive retention efforts
- **Resource Allocation:** Prioritize customer success resources toward high-risk accounts
- **Campaign Targeting:** Target retention campaigns to customers most likely to churn

### Intended Users
- Customer Success Teams
- Marketing Analysts
- Business Intelligence Teams

### Out-of-Scope Uses
- **Credit Decisions:** Not intended for use in credit, lending, or financial risk assessment
- **Employment Decisions:** Not intended for hiring or HR-related decisions
- **Automated Actions:** Should not be used for fully automated customer termination or service denial

---

## Training Data

### Dataset Description
- **Source:** Synthetic customer data (for demonstration)
- **Size:** 1,000 samples
- **Split:** 70% training / 15% validation / 15% test
- **Target Variable:** Binary churn indicator (0 = retained, 1 = churned)
- **Class Balance:** ~20% churn rate (imbalanced)

### Features
| Feature Type | Description |
|-------------|-------------|
| **Demographic** | Customer segment, geographic region |
| **Behavioral** | Usage patterns, engagement metrics |
| **Account** | Tenure, contract type, payment method |
| **Financial** | Monthly charges, total charges |
| **Service** | Number of services, add-ons |

### Preprocessing
- Missing values: Imputed using median (numeric) or mode (categorical)
- Categorical encoding: One-hot encoding
- Numeric scaling: StandardScaler normalization
- Feature selection: All features retained

---

## Evaluation Results

### Performance Metrics (Test Set)

| Metric | Score |
|--------|-------|
| **AUC-ROC** | > 0.75 (target threshold) |
| Accuracy | Varies by model |
| Precision | Varies by model |
| Recall | Varies by model |
| F1 Score | Varies by model |

### Model Comparison

| Model | Validation AUC-ROC |
|-------|-------------------|
| Logistic Regression | Baseline |
| Random Forest | Typically best |
| Gradient Boosting | Competitive |

### Evaluation Methodology
- Stratified train/validation/test split to preserve class distribution
- 5-fold cross-validation during hyperparameter tuning
- Single hold-out test set evaluation for final metrics
- All experiments tracked in MLflow

---

## Limitations

### Known Limitations
1. **Synthetic Data:** Model trained on synthetic data; real-world performance may vary
2. **Class Imbalance:** 20% churn rate may not reflect actual business churn rates
3. **Feature Coverage:** Limited to available customer attributes
4. **Temporal Validity:** Model may degrade over time as customer behavior evolves

### Technical Limitations
- Requires all input features to be present (no missing value handling at inference)
- Predictions are point estimates without uncertainty quantification
- Feature importance based on training data distribution

### Recommendations
- Retrain periodically (quarterly recommended) to maintain performance
- Monitor prediction distribution for data drift
- Validate model performance on new customer cohorts before deployment

---

## Ethical Considerations

### Fairness
- **Protected Attributes:** Model does not explicitly use protected demographic attributes
- **Bias Assessment:** Feature importance should be reviewed to ensure no proxy discrimination
- **Disparate Impact:** Recommend monitoring prediction rates across customer segments

### Privacy
- No personally identifiable information (PII) used in model features
- Customer IDs used only for tracking, not as model features
- Model artifacts do not contain individual customer data

### Transparency
- Model predictions should be explainable to customers upon request
- Feature importance available for interpretation
- Decision thresholds should be clearly documented for operational use

### Human Oversight
- Predictions should inform, not replace, human decision-making
- High-stakes retention actions should involve human review
- Regular model audits recommended

---

## Quantitative Analyses

### Feature Importance
Top contributing features (for tree-based models):
1. Tenure-related features
2. Monthly charges
3. Contract type
4. Payment method
5. Service engagement indicators

### Threshold Analysis
- Default threshold: 0.5
- Consider adjusting based on business cost of false positives vs. false negatives
- Higher threshold = higher precision, lower recall
- Lower threshold = higher recall, lower precision

---

## Caveats and Recommendations

### Deployment Considerations
1. **A/B Testing:** Recommend A/B testing before full deployment
2. **Monitoring:** Implement prediction monitoring for data drift
3. **Feedback Loop:** Capture actual churn outcomes to enable model retraining
4. **Fallback:** Have manual review process for edge cases

### Model Updates
- Retrain when AUC-ROC drops below 0.70 on holdout data
- Update feature engineering as new customer attributes become available
- Document all model changes in version control

---

## MLflow Tracking

### Experiments
- **churn_prediction:** Development experiments
- **churn_prediction_final:** Production model evaluation

### Artifacts
- Model binary (joblib/pickle)
- Preprocessing pipeline
- Feature importance plots
- Confusion matrix visualizations

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2025 | Initial release |

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)

# ML Project Workflow

## Guiding Principles

1. **The Plan is the Source of Truth:** All work must be tracked in `plan.md`
2. **The Tech Stack is Deliberate:** Changes to the tech stack must be documented in `tech-stack.md` *before* implementation
3. **Experiment-Driven Development:** Define hypotheses and success metrics before building models
4. **Reproducibility First:** All experiments must be reproducible with versioned data and code
5. **Model Performance Gates:** Meet defined performance thresholds before deployment
6. **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use `CI=true` for watch-mode tools to ensure single execution.

## Task Workflow

All ML tasks follow a structured lifecycle that balances exploration with rigor.

### Standard ML Task Workflow

1. **Select Task:** Choose the next available task from `plan.md` in sequential order

2. **Mark In Progress:** Before beginning work, edit `plan.md` and change the task from `[ ]` to `[~]`

3. **Define Success Criteria:**
   - Establish baseline metrics (current model performance or naive baseline)
   - Define target metrics and acceptable thresholds
   - Document evaluation methodology
   - **CRITICAL:** Do not proceed without clear, measurable success criteria

4. **Data Preparation:**
   - Validate data quality using tools like Great Expectations, Pandera, or custom checks
   - Document data sources and any preprocessing steps
   - Version datasets using DVC, Delta Lake, or similar
   - Create train/validation/test splits with proper stratification
   - Check for data leakage between splits

5. **Experimentation (Explore Phase):**
   - Log all experiments to tracking system (MLflow, Weights & Biases, Neptune)
   - Record hyperparameters, metrics, and artifacts
   - Use consistent random seeds for reproducibility
   - Document insights and failed approaches
   - **Note:** Multiple iterations are expected - track all experiments

6. **Model Evaluation (Validate Phase):**
   - Compare against baseline metrics
   - Evaluate on held-out test set (only once per experiment cycle)
   - Check for bias/fairness across demographic groups
   - Validate inference latency meets requirements
   - Generate model explanations (SHAP, LIME, etc.) where applicable

7. **Model Documentation:**
   - Create or update model card with:
     - Model description and intended use
     - Training data characteristics
     - Performance metrics across slices
     - Limitations and ethical considerations
     - Version and lineage information

8. **Verify Performance Gates:** Run evaluation suite and verify:
   ```bash
   # Example: Run model evaluation pipeline
   # e.g., python evaluate.py --model models/latest --test-data data/test.csv
   ```
   Target: Meet or exceed defined performance thresholds

9. **Document Deviations:** If implementation differs from tech stack:
   - **STOP** implementation
   - Update `tech-stack.md` with new design
   - Add dated note explaining the change
   - Resume implementation

10. **Commit Code Changes:**
    - Stage all code changes related to the task
    - Propose a clear, concise commit message e.g., `feat(model): Add gradient boosting classifier for churn prediction`
    - Perform the commit

11. **Attach Task Summary with Git Notes:**
    - **Step 11.1: Get Commit Hash:** Obtain the hash of the *just-completed commit* (`git log -1 --format="%H"`)
    - **Step 11.2: Draft Note Content:** Create a detailed summary including:
      - Task name and objective
      - Experiment IDs/links
      - Key metrics achieved
      - Data version used
      - Notable findings or trade-offs
    - **Step 11.3: Attach Note:** Use `git notes add -m "<note content>" <commit_hash>`

12. **Get and Record Task Commit SHA:**
    - **Step 12.1: Update Plan:** Read `plan.md`, find the completed task, update status from `[~]` to `[x]`, and append the first 7 characters of the commit hash
    - **Step 12.2: Write Plan:** Write the updated content back to `plan.md`

13. **Commit Plan Update:**
    - Stage the modified `plan.md` file
    - Commit with message e.g., `conductor(plan): Mark task 'Train churn model' as complete`

### Data Pipeline Task Workflow

For tasks focused on data engineering:

1. **Define Data Contract:** Specify expected schema, data types, and constraints
2. **Implement Pipeline:** Build data transformation logic
3. **Add Data Validation:** Implement quality checks at pipeline stages
4. **Test with Sample Data:** Verify pipeline works on representative samples
5. **Performance Test:** Ensure pipeline handles expected data volumes
6. **Document Lineage:** Record data sources, transformations, and destinations

### Model Deployment Task Workflow

For tasks focused on productionizing models:

1. **Package Model:** Create deployable artifact with dependencies
2. **Write Inference Code:** Implement prediction service/function
3. **Add Input Validation:** Validate inference requests
4. **Implement Monitoring:** Add logging for predictions and drift detection
5. **Load Test:** Verify latency and throughput requirements
6. **Create Rollback Plan:** Document how to revert to previous model version

### Phase Completion Verification and Checkpointing Protocol

**Trigger:** This protocol is executed immediately after a task is completed that also concludes a phase in `plan.md`.

1.  **Announce Protocol Start:** Inform the user that the phase is complete and the verification and checkpointing protocol has begun.

2.  **Ensure Experiment Documentation:**
    -   **Step 2.1: Verify Experiment Tracking:** Confirm all experiments are logged to the tracking system
    -   **Step 2.2: Check Model Artifacts:** Verify model artifacts are versioned and stored
    -   **Step 2.3: Validate Data Versions:** Confirm dataset versions are recorded

3.  **Execute Automated Validation:**
    -   Before execution, announce the exact command to run
    -   **Example:** "I will now run the model evaluation suite. **Command:** `python -m pytest tests/model/ --tb=short`"
    -   Execute the announced command
    -   If validation fails, inform the user and begin debugging (max 2 fix attempts before asking for guidance)

4.  **Propose Manual Verification Plan:**
    -   Generate step-by-step plan based on `product.md` and `plan.md`

    **For a Model Training Phase:**
    ```
    The automated validation has passed. For manual verification:

    **Manual Verification Steps:**
    1.  **Review experiment dashboard:** Open MLflow UI at `http://localhost:5000`
    2.  **Compare to baseline:** Verify the new model's AUC (0.85) exceeds baseline (0.72)
    3.  **Check learning curves:** Confirm no overfitting (training/validation gap < 5%)
    4.  **Review feature importance:** Verify top features align with domain knowledge
    ```

    **For a Data Pipeline Phase:**
    ```
    The automated validation has passed. For manual verification:

    **Manual Verification Steps:**
    1.  **Check data quality report:** Review `reports/data_quality.html`
    2.  **Verify row counts:** Confirm output has expected number of records
    3.  **Spot-check samples:** Review 10 random records for correctness
    ```

5.  **Await Explicit User Feedback:**
    -   Ask: "**Does this meet your expectations? Please confirm with yes or provide feedback.**"
    -   **PAUSE** and await response

6.  **Create Checkpoint Commit:**
    -   Stage all changes
    -   Commit with message e.g., `conductor(checkpoint): Checkpoint end of Phase X`

7.  **Attach Auditable Verification Report using Git Notes:**
    -   Include: validation command, metrics achieved, manual verification steps, user confirmation
    -   Attach to checkpoint commit

8.  **Get and Record Phase Checkpoint SHA:**
    -   Update `plan.md` phase heading with checkpoint hash `[checkpoint: <sha>]`

9.  **Commit Plan Update:**
    -   Commit with message `conductor(plan): Mark phase '<PHASE NAME>' as complete`

10. **Announce Completion:** Inform user that phase and checkpoint are complete

### Quality Gates

Before marking any ML task complete, verify:

**Model Quality:**
- [ ] Model meets performance threshold (accuracy, AUC, F1, etc.)
- [ ] Performance validated on held-out test set
- [ ] No significant performance degradation across data slices
- [ ] Inference latency meets requirements
- [ ] Model size within deployment constraints

**Data Quality:**
- [ ] Data validation checks pass
- [ ] No data leakage between train/test
- [ ] Feature distributions are as expected
- [ ] Missing values handled appropriately

**Reproducibility:**
- [ ] Random seeds set and documented
- [ ] Data version recorded
- [ ] Environment/dependencies pinned
- [ ] Experiment logged to tracking system

**Documentation:**
- [ ] Model card created/updated
- [ ] Experiment notes documented
- [ ] Code follows style guide
- [ ] Key decisions documented

**Ethics & Fairness:**
- [ ] Bias metrics evaluated across groups
- [ ] No discriminatory patterns detected
- [ ] Limitations documented
- [ ] Privacy requirements met

## Development Commands

**AI AGENT INSTRUCTION: Adapt these to the project's specific ML stack.**

### Setup
```bash
# Example: Set up ML development environment
# pip install -r requirements.txt
# dvc pull  # Pull versioned data
# mlflow ui  # Start experiment tracking UI
```

### Daily Development
```bash
# Example: Common ML development tasks
# jupyter lab  # Start notebook server
# python train.py --config configs/experiment.yaml  # Run training
# mlflow runs list --experiment-id 1  # List experiments
# dvc repro  # Reproduce pipeline
```

### Before Committing
```bash
# Example: Pre-commit validation
# python -m pytest tests/
# python -m black src/
# python -m flake8 src/
# dvc status  # Check data version status
```

## Testing Requirements

### Unit Testing
- Test data preprocessing functions
- Test feature engineering logic
- Test model inference functions
- Mock external services (APIs, databases)

### Integration Testing
- Test end-to-end pipeline execution
- Verify model can load and predict
- Test data validation gates
- Check experiment logging

### Model Testing
- Test model performance on known samples
- Verify model handles edge cases gracefully
- Test input validation and error handling
- Benchmark inference latency

## Commit Guidelines

### Message Format
```
<type>(<scope>): <description>

[optional body with experiment details]

[optional footer]
```

### Types
- `feat`: New model, feature, or capability
- `fix`: Bug fix in pipeline or model
- `data`: Data pipeline or preprocessing changes
- `exp`: Experiment-related changes
- `docs`: Documentation updates
- `refactor`: Code restructuring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

### Examples
```bash
git commit -m "feat(model): Add XGBoost classifier for churn prediction"
git commit -m "data(pipeline): Add feature for customer tenure"
git commit -m "exp(tuning): Optimize hyperparameters for gradient boosting"
git commit -m "fix(inference): Handle missing values in prediction input"
```

## Definition of Done

An ML task is complete when:

1. Success criteria defined and documented
2. Model meets performance thresholds
3. All experiments logged to tracking system
4. Data versions recorded
5. Model artifacts versioned and stored
6. Documentation complete (model card, experiment notes)
7. Code passes linting and tests
8. Implementation notes added to `plan.md`
9. Changes committed with proper message
10. Git note with task summary attached

## Emergency Procedures

### Model Performance Degradation in Production
1. Enable shadow mode (new model runs but doesn't serve)
2. Investigate data drift and feature distributions
3. Retrain on recent data if drift detected
4. A/B test before full rollout
5. Document incident and update monitoring

### Data Pipeline Failure
1. Stop downstream processes
2. Identify failure point in pipeline
3. Fix issue and validate with sample data
4. Reprocess affected data batches
5. Verify data quality post-recovery

### Model Bias Detected
1. Take model offline if severe
2. Analyze affected demographic groups
3. Investigate training data for bias sources
4. Implement mitigation (resampling, constraints)
5. Validate fairness metrics before redeployment

## Continuous Improvement

- Review model performance weekly
- Monitor for data and concept drift
- Retrain models on regular schedule
- Update based on user feedback
- Document lessons learned from failed experiments
- Keep models simple and interpretable when possible

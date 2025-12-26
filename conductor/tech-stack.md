# Technology Stack: Customer Churn Prediction

## Overview

This document defines the technology stack for the Customer Churn Prediction project, a Python-based machine learning solution for identifying at-risk customers.

## Programming Language

| Language | Version | Purpose |
|----------|---------|---------|
| Python | 3.x | Primary development language |

## Core Dependencies

### Data Processing
| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0.3 | Data manipulation and analysis |
| numpy | 1.24.3 | Numerical computing and array operations |

### Machine Learning
| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3.0 | ML algorithms, model training, and evaluation |

### Visualization
| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | 3.7.2 | Data visualization and plotting |

### Development Environment
| Package | Version | Purpose |
|---------|---------|---------|
| jupyter | 1.0.0 | Interactive notebook development |

### ML Infrastructure
| Package | Version | Purpose |
|---------|---------|---------|
| mlflow | 2.5.0 | Experiment tracking, model registry, and reproducibility |

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Version Management

- Dependencies are pinned to specific versions in `requirements.txt`
- Update versions deliberately after testing compatibility
- Use MLflow to track which dependency versions produced each model

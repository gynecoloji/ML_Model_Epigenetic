# Gene Expression Prediction using Stacking Ensemble

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20Stacking-orange.svg)](https://scikit-learn.org/)

A comprehensive machine learning pipeline for predicting gene expression levels from ChIP-seq features using a two-level stacking ensemble approach with rigorous gene-based cross-validation.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contact](#contact)

## 🔬 Overview

This project implements a **two-level stacking ensemble** for gene expression prediction, combining multiple base learners through a meta-model. The pipeline emphasizes:

- **Gene-based data splitting** to prevent data leakage across cell lines
- **Comprehensive hyperparameter tuning** for both base models and meta-model
- **Out-of-fold predictions** for proper stacking without overfitting
- **Multi-metric evaluation** (R², RMSE, MAE, Pearson r, Spearman ρ)

## ✨ Key Features

### Data Handling
- ✅ Gene-based train/test splitting using `GroupShuffleSplit` (80/20 split)
- ✅ Prevents data leakage: no gene appears in both train and test sets
- ✅ Supports multiple cell lines in the same dataset

### Model Training
- ✅ **5 diverse base models**: Random Forest, XGBoost, LightGBM, ElasticNet, Ridge
- ✅ Automated hyperparameter tuning with `RandomizedSearchCV` (20 iterations per model)
- ✅ Gene-based cross-validation with `GroupKFold` (5 folds)
- ✅ Ridge regression meta-model with `GridSearchCV` tuning (9 alpha values)

### Evaluation & Output
- ✅ Comprehensive performance metrics on test set
- ✅ Individual model evaluation and comparison
- ✅ Per-cell-line performance breakdown
- ✅ Automatic saving of models, predictions, and results
- ✅ Detailed experiment documentation and README generation

## 🧪 Methodology

### Two-Level Stacking Architecture
```
Level 0 (Base Models)           Level 1 (Meta-Model)
┌─────────────────┐
│ Random Forest   │──┐
├─────────────────┤  │
│ XGBoost         │──┤
├─────────────────┤  ├──► Out-of-Fold ────► Ridge Meta-Model ────► Final Prediction
│ LightGBM        │──┤    Predictions
├─────────────────┤  │
│ ElasticNet      │──┤
├─────────────────┤  │
│ Ridge           │──┘
└─────────────────┘
```

### Cross-Validation Strategy

The pipeline uses **GroupKFold** with gene-based splitting:

1. **Train/Test Split**: 80/20 split ensuring no gene overlap
2. **5-Fold CV**: Each fold respects gene boundaries
3. **Out-of-Fold Predictions**: Used to train meta-model without data leakage
4. **Meta-Model Tuning**: GridSearchCV on out-of-fold predictions

This approach ensures:
- No information leakage between folds
- Realistic performance estimates
- Proper generalization to unseen genes

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages
```bash
pip install pandas numpy scikit-learn xgboost lightgbm scipy joblib
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
scipy>=1.7.0
joblib>=1.1.0
```

## 💻 Usage

### Basic Usage
```python
# Run the complete pipeline
python stacking_ensemble.py
```

### Expected Input Data Format

The pipeline expects a CSV file at `models/data/combined_data.csv` with the following structure:

| Column | Description |
|--------|-------------|
| `gene_id` | Unique gene identifier |
| `cell_line` | Cell line name |
| `expression` | Target variable (gene expression level) |
| `feature_1, feature_2, ...` | ChIP-seq features |

### Loading Trained Models
```python
import joblib
import numpy as np
import pandas as pd

# Load meta-model
meta_model = joblib.load('models/results/stacking/[experiment_name]/meta_model.pkl')

# Load base models
base_models = {}
model_names = ['RandomForest', 'XGBoost', 'LightGBM', 'ElasticNet', 'Ridge']
for name in model_names:
    base_models[name] = joblib.load(f'models/results/stacking/[experiment_name]/base_models/{name}.pkl')

# Make predictions on new data
def predict_ensemble(X_new):
    # Get base model predictions
    base_preds = np.zeros((len(X_new), len(base_models)))
    for i, model in enumerate(base_models.values()):
        base_preds[:, i] = model.predict(X_new)
    
    # Get final predictions from meta-model
    final_preds = meta_model.predict(base_preds)
    return final_preds

# Example usage
X_new = pd.read_csv('new_data.csv')
predictions = predict_ensemble(X_new)
```

## 📁 Project Structure
```
ML/
├── scripts/
│   ├──stacking.py          # Main pipeline script
├── models/
│   ├── data/
│   │   └── combined_data.csv     # Input dataset
│   └── results/
│       └── stacking/
│           └── stacking_ensemble_[timestamp]/
│               ├── base_models/
│               │   ├── RandomForest.pkl
│               │   ├── XGBoost.pkl
│               │   ├── LightGBM.pkl
│               │   ├── ElasticNet.pkl
│               │   └── Ridge.pkl
│               ├── meta_model.pkl
│               ├── oof_predictions.npy
│               ├── test_base_predictions.npy
│               ├── stacked_predictions.npy
│               ├── test_predictions_detailed.csv
│               ├── performance_summary.csv
│               ├── performance_metrics.json
│               ├── experiment_config.json
│               ├── best_parameters.json
│               ├── tuning_results.pkl
│               ├── train_test_split.json
│               └── README.md
└── Stacking_README.md
```

## 🏗️ Model Architecture

### Base Models (Level 0)

| Model | Key Hyperparameters Tuned | Search Space |
|-------|---------------------------|--------------|
| **Random Forest** | n_estimators, max_depth, min_samples_split | 4×5×4 combinations |
| **XGBoost** | n_estimators, max_depth, learning_rate, reg_alpha | 4×4×4×4 combinations |
| **LightGBM** | n_estimators, max_depth, learning_rate, reg_lambda | 4×4×4×3 combinations |
| **ElasticNet** | alpha, l1_ratio | 5×5 combinations |
| **Ridge** | alpha | 6 values |

### Meta-Model (Level 1)

- **Type**: Ridge Regression
- **Tuning**: GridSearchCV with 9 alpha values [0.001 to 100]
- **Input**: 5 features (predictions from each base model)
- **Output**: Final gene expression prediction

## 📊 Results

### Performance Metrics

The pipeline evaluates models using:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Pearson r**: Pearson correlation coefficient
- **Spearman ρ**: Spearman rank correlation

### Output Files

1. **performance_summary.csv**: Comparative table of all models
2. **performance_metrics.json**: Detailed metrics including per-cell-line performance
3. **test_predictions_detailed.csv**: Complete predictions with residuals for all models

### Expected Improvement

Stacking ensembles typically achieve **2-8% improvement** in R² over the best individual base model, depending on:
- Diversity of base models
- Quality of out-of-fold predictions
- Optimal meta-model regularization

## 🔧 Customization

### Adding New Base Models
```python
# In SECTION 2, add to base_models dictionary
base_models['NewModel'] = YourModelClass(
    param1=value1,
    param2=value2,
    random_state=42
)

# Add corresponding hyperparameter search space
param_distributions['NewModel'] = {
    'param1': [value1, value2, value3],
    'param2': [value_a, value_b, value_c]
}
```

### Modifying Cross-Validation
```python
# Change number of folds
n_folds = 5  # Default is 5

# Use different CV strategy (must support groups)
from sklearn.model_selection import StratifiedGroupKFold
gkf = StratifiedGroupKFold(n_splits=n_folds)
```

## 📧 Contact

**Author**: Ji  
**Email**: gynecoloji@gmail.com  
**GitHub**: [@gynecoloji](https://github.com/gynecoloji)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn for ML framework
- XGBoost and LightGBM teams for gradient boosting implementations
- The bioinformatics community for ChIP-seq analysis methods

**Last Updated**: January 2026  
**Version**: 1.0.0
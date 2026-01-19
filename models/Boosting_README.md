# Gene Expression Prediction using Boosting Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)

A comprehensive machine learning pipeline for predicting gene expression levels from ChIP-seq features using ensemble boosting methods with rigorous cross-validation.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Structure](#data-structure)
- [Methodology](#methodology)
- [Usage](#usage)
- [Outputs](#outputs)
- [Model Configurations](#model-configurations)
- [Results Interpretation](#results-interpretation)
- [Contact](#contact)

## 🎯 Overview

This project implements and compares **9 different boosting model configurations** across three algorithms:
- **Gradient Boosting** (sklearn)
- **XGBoost**
- **LightGBM**

Each algorithm is tested with three regularization strategies (Conservative, Moderate, Aggressive) to predict gene expression levels from ChIP-seq bin features across multiple cell lines.

### Key Highlights
- ✅ **Gene-based splitting**: Prevents data leakage by keeping all observations from the same gene together
- ✅ **5-fold cross-validation**: Robust performance estimation using GroupKFold
- ✅ **Comprehensive evaluation**: Multiple metrics (R², RMSE, MAE, Pearson, Spearman correlations)
- ✅ **Cell line analysis**: Per-cell line performance assessment
- ✅ **Rich visualizations**: 5 detailed figures for model comparison
- ✅ **Automatic model selection**: Saves best performing models

## ✨ Features

- **9 Boosting Configurations**: Systematic comparison of model complexity levels
- **Gene-Level Splitting**: 80/20 train-test split ensuring no gene appears in both sets
- **5-Fold Cross-Validation**: Gene-based GroupKFold for reliable performance estimates
- **Multi-Metric Evaluation**: R², RMSE, MAE, Pearson R, Spearman R
- **Cell Line Consistency**: Performance tracking across different cell lines
- **Generalization Assessment**: Overfitting detection via train-test gap analysis
- **Composite Scoring**: Multi-criteria model ranking
- **Automated Reporting**: CSV results, summary reports, and visualizations

## 🛠️ Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📥 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/gynecoloji/gene-expression-prediction.git
cd gene-expression-prediction
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Data Structure

### Input Data Format
The code expects a CSV file with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `gene_id` | str | Unique gene identifier |
| `cell_line` | str | Cell line name (e.g., K562, GM12878, etc.) |
| `expression` | float | Target expression level |
| `bin_1`, `bin_2`, ... | float | ChIP-seq feature bins (feature columns) |

### Example:
```csv
gene_id,cell_line,expression,bin_1,bin_2,bin_3,...
ENSG001,K562,5.23,0.12,0.45,0.78,...
ENSG001,GM12878,4.87,0.15,0.42,0.71,...
ENSG002,K562,3.45,0.08,0.33,0.56,...
```

### Data Location
Place your data file at:
```
ML/models/data/combined_data.csv
```

## 🔬 Methodology

### 1. Data Preparation
- Extract features (ChIP-seq bins) and target (expression)
- Identify unique genes and cell lines
- Check for missing values

### 2. Gene-Based Splitting
- **Train Set**: 80% of genes (all their observations)
- **Test Set**: 20% of genes (all their observations)
- **Validation**: Zero gene overlap between sets

### 3. Cross-Validation Strategy
- **Method**: 5-Fold GroupKFold
- **Grouping**: By gene_id
- **Purpose**: Reliable performance estimation on training data

### 4. Model Training
For each of 9 configurations:
1. Perform 5-fold CV on training set
2. Train final model on full training set
3. Evaluate on held-out test set
4. Assess per-cell line performance

### 5. Evaluation Metrics
- **R² Score**: Variance explained
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **Pearson R**: Linear correlation
- **Spearman R**: Rank correlation
- **Generalization**: Train-test gap
- **Consistency**: Std across cell lines

## 🚀 Usage

### Basic Usage
```python
# Ensure you're in the correct directory
import os
os.chdir("ML")

# Run the complete pipeline
python boosting_model_pipeline.py
```

### Step-by-Step Execution

**Step 1: Load and inspect data**
```python
import pandas as pd
df = pd.read_csv("models/data/combined_data.csv")
print(f"Dataset shape: {df.shape}")
print(f"Unique genes: {df['gene_id'].nunique()}")
print(f"Cell lines: {df['cell_line'].unique()}")
```

**Step 2: Prepare train-test split**
```python
from sklearn.model_selection import GroupShuffleSplit

# Extract features
feature_cols = [col for col in df.columns 
                if col not in ['gene_id', 'cell_line', 'expression']]
X = df[feature_cols].values
y = df['expression'].values
groups = df['gene_id'].values

# Split with gene-based grouping
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
```

**Step 3: Run cross-validation**
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
cv = GroupKFold(n_splits=5)

# CV on training set only
for train_fold, val_fold in cv.split(X[train_idx], y[train_idx], groups[train_idx]):
    model.fit(X[train_idx][train_fold], y[train_idx][train_fold])
    score = model.score(X[train_idx][val_fold], y[train_idx][val_fold])
    print(f"Fold R²: {score:.4f}")
```

## 📂 Outputs

### Directory Structure
```
ML/
├── models/
│   ├── results/
│   │   └── boosting/
│   │       ├── boosting_configurations_results.csv
│   │       ├── boosting_per_cell_line_results.csv
│   │       ├── model_type_comparison.csv
│   │       ├── boosting_summary_report.txt
│   │       ├── all_configurations.pkl
│   │       └── split_info.pkl
│   └── trained/
│       ├── best_boosting_model.pkl
│       ├── best_model_metadata.pkl
│       ├── top1_*.pkl
│       ├── top2_*.pkl
│       └── top3_*.pkl
└── analysis/
    └── figures/
        └── boosting/
            ├── 01_performance_dashboard.png
            ├── 02_model_type_comparison.png
            ├── 03_cell_line_heatmaps.png
            ├── 04_cell_line_analysis.png
            └── 05_multi_criteria_ranking.png
```

### Output Files Description

#### Results Files
| File | Description |
|------|-------------|
| `boosting_configurations_results.csv` | Complete metrics for all 9 configurations with CV results |
| `boosting_per_cell_line_results.csv` | Performance breakdown by cell line for each config |
| `model_type_comparison.csv` | Statistical comparison of GB, XGB, and LGBM |
| `boosting_summary_report.txt` | Human-readable comprehensive report |

#### Model Files
| File | Description |
|------|-------------|
| `best_boosting_model.pkl` | Best model (highest composite score) |
| `best_model_metadata.pkl` | Model parameters and performance metrics |
| `top1_*.pkl`, `top2_*.pkl`, `top3_*.pkl` | Top 3 performing models |

#### Visualization Files
| File | Content |
|------|---------|
| `01_performance_dashboard.png` | 3×3 grid: R², correlation, generalization, cell line consistency |
| `02_model_type_comparison.png` | GB vs XGB vs LGBM comparison |
| `03_cell_line_heatmaps.png` | R² and Pearson correlation by cell line |
| `04_cell_line_analysis.png` | Cell line distribution and consistency analysis |
| `05_multi_criteria_ranking.png` | Multi-metric performance comparison |

## ⚙️ Model Configurations

### Gradient Boosting (sklearn)
```python
'GradientBoosting_Conservative': {
    'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3,
    'min_samples_split': 50, 'subsample': 0.8
}
'GradientBoosting_Moderate': {
    'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 4,
    'min_samples_split': 30, 'subsample': 0.8
}
'GradientBoosting_Aggressive': {
    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5,
    'min_samples_split': 20, 'subsample': 0.8
}
```

### XGBoost
```python
'XGBoost_Conservative': {
    'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3,
    'reg_alpha': 0.1, 'reg_lambda': 1.0
}
'XGBoost_Moderate': {
    'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 4,
    'reg_alpha': 0.05, 'reg_lambda': 0.5
}
'XGBoost_Aggressive': {
    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5,
    'reg_alpha': 0.01, 'reg_lambda': 0.1
}
```

### LightGBM
```python
'LightGBM_Conservative': {
    'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 15,
    'reg_alpha': 0.1, 'reg_lambda': 1.0
}
'LightGBM_Moderate': {
    'n_estimators': 150, 'learning_rate': 0.1, 'num_leaves': 31,
    'reg_alpha': 0.05, 'reg_lambda': 0.5
}
'LightGBM_Aggressive': {
    'n_estimators': 200, 'learning_rate': 0.1, 'num_leaves': 50,
    'reg_alpha': 0.01, 'reg_lambda': 0.1
}
```

## 📈 Results Interpretation

### Key Metrics to Monitor

1. **Test R²**: Primary performance metric (higher is better)
   - > 0.7: Excellent
   - 0.5-0.7: Good
   - < 0.5: Needs improvement

2. **R² Gap** (Train R² - Test R²): Overfitting indicator
   - < 0.15: Good generalization
   - 0.15-0.25: Moderate overfitting
   - > 0.25: Significant overfitting

3. **CV R² Std**: Cross-validation stability
   - < 0.05: Very stable
   - 0.05-0.10: Moderately stable
   - > 0.10: Unstable

4. **R² Std Across Cell Lines**: Model consistency
   - < 0.1: Highly consistent
   - 0.1-0.2: Moderately consistent
   - > 0.2: Inconsistent

5. **Composite Score**: Overall ranking (0-1, higher is better)
   - Weights: Test R² (35%), Pearson (25%), Generalization (20%), Consistency (15%), Efficiency (5%)

### Best Performer Selection

The pipeline automatically selects the best model based on:
```python
composite_score = (0.35 * test_r2 + 
                   0.25 * test_pearson +
                   0.20 * generalization_score + 
                   0.15 * consistency_score + 
                   0.05 * efficiency_score)
```

## 🔍 Example Results

Typical output summary:
```
Best Performing Configuration: LightGBM_Moderate
  Test R²: 0.6234
  CV R²: 0.6187 ± 0.0423
  Test Pearson: 0.7891
  R² Gap: 0.1234
  Composite Score: 0.6453
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Author**: [Your Name]  
**Email**: gynecoloji@gmail.com  
**GitHub**: [https://github.com/gynecoloji](https://github.com/gynecoloji?tab=repositories)

## 🙏 Acknowledgments

- **scikit-learn** for machine learning infrastructure
- **XGBoost** and **LightGBM** teams for gradient boosting implementations
- The bioinformatics community for ChIP-seq methodology

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{gene_expression_boosting,
  author = {Your Name},
  title = {Gene Expression Prediction using Boosting Models},
  year = {2025},
  url = {https://github.com/gynecoloji/gene-expression-prediction}
}
```

## 🔄 Version History

- **v1.0.0** (2025-01-11): Initial release with 9 boosting configurations, 5-fold CV, and comprehensive evaluation

---

**Last Updated**: January 2025  
**Status**: Active Development

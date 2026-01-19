# Gene Expression Prediction using Bagging Ensemble Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning pipeline for predicting gene expression levels from ChIP-seq features using Bagging ensemble models with gene-based cross-validation.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Model Configurations](#model-configurations)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Results Interpretation](#results-interpretation)
- [Contact](#contact)

## 🔬 Overview

This project implements a robust machine learning framework for predicting gene expression from ChIP-seq data. It uses **Bagging (Bootstrap Aggregating)** ensemble methods with Decision Trees as base estimators, ensuring no gene leakage between train and test sets through gene-based data splitting.

### Key Objectives
- Predict gene expression levels across multiple cell lines
- Prevent data leakage through gene-based train-test splitting
- Compare multiple model configurations systematically
- Evaluate model performance across different cell lines
- Select the best model based on composite scoring

## ✨ Features

- **Gene-Based Data Splitting**: Ensures all observations from the same gene stay in the same split (train or test)
- **Multiple Configurations**: Tests 7 different hyperparameter combinations ranging from conservative to less regularized
- **Comprehensive Evaluation**: 
  - R² score, RMSE, MAE
  - Pearson and Spearman correlations
  - Per-cell-line analysis
  - Generalization metrics
- **Rich Visualizations**: 
  - Performance dashboards
  - Cell line heatmaps
  - Consistency analysis
  - Multi-criteria rankings
- **Automated Model Selection**: Composite scoring to identify the optimal configuration
- **Full Reproducibility**: Saves models, splits, and metadata for future use

## 📁 Project Structure

```
ML/
├── models/
│   ├── data/
│   │   └── combined_data.csv          # Input dataset
│   ├── results/
│   │   └── bagging/
│   │       ├── bagging_configurations_results.csv
│   │       ├── bagging_per_cell_line_results.csv
│   │       ├── bagging_summary_report.txt
│   │       ├── train_test_split.pkl
│   │       └── all_configurations.pkl
│   └── trained/
│       ├── best_bagging_model.pkl     # Best trained model
│       └── best_model_metadata.pkl    # Model metadata
├── analysis/
│   └── figures/
│       └── bagging/
│           ├── 01_comprehensive_dashboard.png
│           ├── 02_cell_line_heatmaps.png
│           ├── 03_cell_line_analysis.png
│           └── 04_multi_criteria_ranking.png
└── main_script.py                      # Main analysis script
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Packages

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn scipy
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Requirements.txt
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## 📊 Data Requirements

### Input Format
The script expects a CSV file (`combined_data.csv`) with the following structure:

| Column | Description | Type |
|--------|-------------|------|
| `gene_id` | Unique gene identifier | String/Integer |
| `cell_line` | Cell line name | String |
| `expression` | Target variable (gene expression level) | Float |
| `bin_1`, `bin_2`, ... | ChIP-seq feature bins | Float |

### Data Characteristics
- **Observations**: Multiple cell lines per gene
- **Features**: ChIP-seq bins (numerical features)
- **Target**: Continuous expression values
- **No missing values** in expression or features

## 🚀 Usage

### Basic Usage

1. **Prepare your data**:
   - Place your `combined_data.csv` file in `ML/models/data/`
   - Ensure the data follows the required format

2. **Run the analysis**:
   ```bash
   cd ML
   python main_script.py
   ```

3. **Check outputs**:
   - Results: `models/results/bagging/`
   - Figures: `analysis/figures/bagging/`
   - Trained model: `models/trained/`

### Workflow Steps

The script automatically performs the following steps:

1. **Data Loading & Exploration**
   - Loads dataset
   - Displays data structure
   - Checks for missing values
   - Verifies gene and cell line distributions

2. **Gene-Based Train-Test Split**
   - 80/20 train-test split
   - Ensures no gene appears in both sets
   - Maintains cell line distributions

3. **Model Training & Evaluation**
   - Tests 7 configurations
   - Trains Bagging models with different hyperparameters
   - Evaluates on multiple metrics
   - Analyzes per-cell-line performance

4. **Visualization**
   - Creates comprehensive performance dashboards
   - Generates heatmaps for cell line analysis
   - Plots consistency metrics
   - Produces multi-criteria rankings

5. **Results Saving**
   - Saves all results to CSV
   - Exports best model (pickle)
   - Creates summary report
   - Stores metadata for reproducibility

## ⚙️ Model Configurations

The script tests 7 different configurations:

| Configuration | Max Depth | Min Samples Split | Min Samples Leaf | N Estimators | Max Samples | Description |
|---------------|-----------|-------------------|------------------|--------------|-------------|-------------|
| Config_1 | 6 | 150 | 75 | 100 | 0.7 | Very Conservative |
| Config_2 | 8 | 100 | 50 | 100 | 0.8 | Conservative |
| Config_3 | 10 | 50 | 25 | 100 | 0.8 | Moderate |
| Config_4 | 12 | 30 | 15 | 100 | 0.7 | Balanced |
| Config_5 | 15 | 20 | 10 | 100 | 0.8 | Less Regularized |
| Config_6 | 10 | 50 | 25 | 150 | 0.7 | More Trees |
| Config_7 | 20 | 40 | 20 | 80 | 0.75 | Deep with Constraints |

### How Configurations Differ
- **Conservative configs** (1-2): High regularization, shallow trees → Less overfitting
- **Moderate configs** (3-4): Balanced complexity → Good generalization
- **Aggressive configs** (5-7): Lower regularization → Higher capacity

## 📈 Evaluation Metrics

### Overall Performance
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

### Correlation Metrics
- **Pearson R**: Linear correlation (-1 to 1)
- **Spearman R**: Rank correlation (-1 to 1)

### Generalization Metrics
- **R² Gap**: Difference between train and test R² (lower is better)
- **R² Ratio**: Test R² / Train R² (closer to 1 is better)
- **Generalization Score**: 1 - |R² Gap| (closer to 1 is better)

### Cell Line Consistency
- **R² Std**: Standard deviation of R² across cell lines (lower = more consistent)
- **Min/Max R²**: Range of performance across cell lines

### Composite Score
Weighted combination of:
- 35% Test R²
- 25% Test Pearson
- 20% Low R² Gap
- 15% Cell Line Consistency
- 5% Training Efficiency

## 📤 Output Files

### 1. Results CSVs
- **`bagging_configurations_results.csv`**: All configuration metrics
- **`bagging_per_cell_line_results.csv`**: Per-cell-line performance details

### 2. Model Files
- **`best_bagging_model.pkl`**: Trained best model (pickle format)
- **`best_model_metadata.pkl`**: Model parameters and metadata

### 3. Reproducibility Files
- **`train_test_split.pkl`**: Train/test indices and gene lists
- **`all_configurations.pkl`**: All tested hyperparameter configurations

### 4. Reports
- **`bagging_summary_report.txt`**: Human-readable summary with top performers

### 5. Visualizations
- **`01_comprehensive_dashboard.png`**: 9-panel performance overview
- **`02_cell_line_heatmaps.png`**: R² and Pearson heatmaps by cell line
- **`03_cell_line_analysis.png`**: Box plots and trajectory analysis
- **`04_multi_criteria_ranking.png`**: Multi-metric comparison

## 🎯 Results Interpretation

### What to Look For

1. **Best Overall Model**
   - Check the configuration with highest Composite Score
   - Verify good Test R² (typically > 0.5)
   - Ensure low R² Gap (< 0.15 is good, < 0.25 is acceptable)

2. **Generalization Quality**
   - R² Gap < 0.15: Excellent generalization
   - R² Gap 0.15-0.25: Good generalization
   - R² Gap > 0.25: May be overfitting

3. **Cell Line Consistency**
   - Low R² Std across cell lines indicates robust model
   - Check heatmaps for any poorly-performing cell lines

4. **Correlation Analysis**
   - High Pearson R: Strong linear relationship
   - High Spearman R: Good rank ordering (monotonic relationship)
   - If Spearman >> Pearson: Non-linear relationship present

### Example Interpretation

```
✓ Best Model: Config_4_Balanced
  - Test R²: 0.582 (explains 58% of variance)
  - Test Pearson: 0.764 (strong linear correlation)
  - R² Gap: 0.148 (good generalization)
  - R² Std: 0.042 (consistent across cell lines)
  - Composite Score: 0.685
  
→ This model shows strong performance with good generalization
→ Performs consistently across all cell lines
→ Recommended for production use
```

## 🔄 Customization

### Adding New Configurations

Edit the `param_configs` dictionary:

```python
param_configs = {
    'Config_8_Custom': {
        'base': {
            'max_depth': 14,
            'min_samples_split': 40,
            'min_samples_leaf': 20,
            'max_features': 0.75
        },
        'bagging': {
            'n_estimators': 120,
            'max_samples': 0.75,
            'max_features': 0.85
        }
    },
    # ... existing configs
}
```

### Modifying Evaluation Metrics

Adjust the composite score weights:

```python
composite_score_2 = (
    0.35 * norm_test_r2 +      # Test R² weight
    0.25 * norm_test_pearson + # Pearson weight
    0.20 * norm_gap +          # Generalization weight
    0.15 * norm_consistency +  # Consistency weight
    0.05 * norm_efficiency     # Efficiency weight
)
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: combined_data.csv`
- **Solution**: Ensure data file is in `ML/models/data/` directory

**Issue**: Gene leakage detected
- **Solution**: Verify gene IDs are consistent and no duplicates exist

**Issue**: Low R² scores (< 0.3)
- **Solution**: Check data quality, feature engineering, or try different model types

**Issue**: High R² gap (> 0.3)
- **Solution**: Increase regularization (lower max_depth, higher min_samples_split)

**Issue**: Memory errors
- **Solution**: Reduce n_estimators or use fewer configurations

## 📚 References

### Key Papers
- Breiman, L. (1996). "Bagging Predictors". *Machine Learning*, 24(2), 123-140.
- Friedman, J., Hastie, T., & Tibshirani, R. (2001). *The Elements of Statistical Learning*.

### Documentation
- [scikit-learn Bagging](https://scikit-learn.org/stable/modules/ensemble.html#bagging)
- [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)

## 📧 Contact

**Author**: [Your Name]  
**Email**: gynecoloji@gmail.com  
**GitHub**: [https://github.com/gynecoloji](https://github.com/gynecoloji?tab=repositories)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn team for excellent ML tools
- All contributors and users of this project

---

**Last Updated**: January 2026  
**Version**: 1.0.0

---

## 🚀 Quick Start Example

```python
# Load the saved best model
import pickle

with open('models/trained/best_bagging_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions on new data
predictions = model.predict(X_new)

# Load metadata
with open('models/trained/best_model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Model Test R²: {metadata['test_r2']:.4f}")
print(f"Configuration: {metadata['configuration_name']}")
```

---

*For questions or issues, please open an issue on GitHub or contact via email.*

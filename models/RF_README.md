# Random Forest Model for Gene Expression Prediction from Histone Modifications

This pipeline trains and analyzes a Random Forest model to predict RNA gene expression levels from histone modification patterns (ChIP-seq data).

## Overview

**Goal**: Predict gene expression from histone mark signals across genomic regions (5% upstream + gene body + 5% downstream)

**Input Data**: `models/data/combined_data.csv` containing:
- Histone mark signals binned across gene regions (features ending with `_bin[N]`)
- Gene expression levels (`expression` column)
- Gene identifiers (`gene_id` column)

**Model**: Random Forest Regressor with hyperparameter tuning via GroupKFold cross-validation

---

## Pipeline Steps

### 1. **Hyperparameter Tuning** (`tune_rf_with_cv`)
- **What**: Grid search over Random Forest parameters using 5-fold GroupKFold CV
- **Why GroupKFold**: Prevents gene leakage - ensures all samples from the same gene stay in the same fold
- **Parameters tested**:
  - `n_estimators`: [50, 100, 200, 300]
  - `max_depth`: [None, 20, 30, 40]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2']
- **Output**: Best model, best parameters, CV results

### 2. **Model Assessment** (`assess_rf_model`)
Comprehensive evaluation on both training and test sets:
- **Metrics**: R², RMSE, MAE, Pearson/Spearman correlations, MAPE
- **Visualizations**:
  - Prediction scatter plots (predicted vs true expression)
  - Residual analysis (residuals vs predicted, residual distributions)
- **Overfitting analysis**: Compares train vs test performance

### 3. **Feature Importance Analysis** (`analyze_rf_feature_importance`)
Analyzes which histone marks and genomic regions drive predictions:
- **Global importance**: Overall contribution of each histone mark
- **Regional importance**: Which gene regions matter most (upstream, TSS, gene body, downstream)
- **Bin-level importance**: Spatial patterns across gene regions

### 4. **SHAP Analysis** (Multiple Functions)
Advanced model interpretation showing **how** features impact predictions:

#### 4A. **SHAP Calculation** (`calculate_shap_values_parallel`)
- Parallel processing for fast computation (uses all CPU cores)
- TreeExplainer optimized for Random Forests
- Outputs SHAP values for every feature and prediction

#### 4B. **SHAP Visualizations**
- **Summary plots** (`plot_shap_summary`): Feature importance + impact direction
- **Dependence plots**: How feature values affect predictions
- **Waterfall plots**: Explain individual predictions
- **Regional analysis** (`analyze_shap_by_region_sign`): Impact by genomic region

#### 4C. **Signed SHAP Analysis**
- **With sign preserved**: Shows which features increase vs decrease expression
- **Diverging colormaps**: Blue = decreases expression, Red = increases expression
- **Positive/Negative breakdown**: Quantifies bidirectional effects

---

## Output Files

### Model Files (`models/trained/`)
```
rf_model.pkl                    # Trained Random Forest model (joblib)
```

### Results (`models/results/rf/`)
```
rf_cv_results.csv               # Full cross-validation results
rf_metrics_comparison.csv       # Train vs test metrics
rf_feature_importances.csv      # Feature importance values
rf_importance_matrix.csv        # Histone mark × bin importance matrix
rf_histone_mark_importance.csv  # Overall histone mark importance
rf_regional_importance.csv      # Importance by genomic region

shap_values.csv                 # SHAP values for all test samples
shap_matrix_signed.csv          # Mean SHAP by histone mark and bin (with sign)
regional_shap_signed.csv        # Regional SHAP impact (with sign)
```

### Predictions (`analysis/figures/rf/`)
```
rf_train_predictions.csv        # Training set predictions
rf_test_predictions.csv         # Test set predictions
```

### Visualizations (`analysis/figures/rf/`)

#### Model Performance
```
rf_prediction_scatter.png       # Predicted vs true expression
rf_residual_analysis.png        # Residual plots
```

#### Feature Importance
```
rf_importance_heatmap.png       # Heatmap across gene regions
rf_importance_profiles.png      # Profiles for each histone mark
rf_histone_importance.png       # Overall histone mark comparison
rf_regional_importance.png      # Regional contribution comparison
rf_importance_summary.png       # Comprehensive summary figure
```

#### SHAP Analysis
```
shap_beeswarm.png              # Top 20 features (importance + direction)
shap_bar.png                   # Mean absolute SHAP values
shap_heatmap_signed.png        # Signed impact across gene regions
shap_profiles_signed.png       # Signed profiles per histone mark
regional_shap_signed.png       # Regional impact (signed)
shap_waterfall_sample_*.png    # Individual prediction explanations
```

---

## Usage

### Basic Usage
```python
# The pipeline runs automatically when you execute the script
python rf_analysis.py
```

### Key Parameters to Adjust

#### Computational Resources
```python
# In tune_rf_with_cv()
n_jobs=24                      # Set to your CPU core count

# In calculate_shap_values_parallel()
n_jobs=24                      # Set to your CPU core count
max_samples=1000               # Background samples for SHAP (higher = more accurate but slower)
```

#### Cross-Validation
```python
# In tune_rf_with_cv()
cv_folds=5                     # Number of CV folds (higher = better but slower)
```

#### Hyperparameter Grid
```python
# Modify param_grid in tune_rf_with_cv() to test different values
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 20, 30, 40],
    # ... add or modify parameters
}
```

---

## Key Features

### 1. **Gene-Aware Splitting**
- Uses `GroupShuffleSplit` and `GroupKFold` to prevent gene leakage
- Ensures all samples from the same gene stay together in train or test
- Critical for unbiased evaluation

### 2. **Parallel Processing**
- Hyperparameter tuning: `n_jobs=-1` uses all cores
- SHAP calculation: Parallelized across test samples
- Significantly faster than serial processing

### 3. **Comprehensive Interpretation**
- **Built-in importance**: Which features are used most by the model
- **SHAP values**: How features impact individual predictions (with direction)
- **Regional analysis**: Which parts of genes matter most

### 4. **Signed SHAP Analysis**
- Preserves direction of impact (positive = increases expression, negative = decreases)
- Uses diverging colormaps for intuitive visualization
- Reveals activating vs repressive histone marks

---

## Understanding the Outputs

### Feature Importance vs SHAP

| Metric | What it shows | Use case |
|--------|--------------|----------|
| **Feature Importance** | How much each feature is used in splits | Which features are important for the model |
| **SHAP (absolute)** | Average magnitude of impact | Which features strongly affect predictions (either direction) |
| **SHAP (signed)** | Direction and magnitude of impact | Which features increase vs decrease expression |

### Region Definitions
Based on gene structure (110% total = 5% upstream + 100% gene body + 5% downstream):
- **Upstream** (bins 1 to ~5): Promoter region
- **TSS Region** (0%-10%): Transcription start site vicinity
- **Gene Body** (10%-100%): Gene coding region
  - Split into thirds for finer analysis: 10-33%, 33-66%, 66-100%
- **Downstream** (bins ~105 to 110): Termination region

---

## Expected Runtime

For a typical dataset with ~10,000 samples, 5 histone marks, 1000 bins:

| Step | Approximate Time |
|------|-----------------|
| Hyperparameter tuning | 30-60 minutes |
| Model assessment | 1-2 minutes |
| Feature importance analysis | <1 minute |
| SHAP calculation (parallel, 24 cores) | 5-10 minutes |
| Visualization generation | 2-3 minutes |
| **Total** | **45-90 minutes** |

*Note: Times vary based on dataset size, CPU cores, and hyperparameter grid*

---

## Contact

For questions or issues with this pipeline, please contact gynecoloji (gynecoloji@gmail.com).

---

## License

MIT
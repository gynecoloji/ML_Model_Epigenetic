# Simple Linear Regression Model (SLR)

## Overview
This folder contains scripts and results for Simple Linear Regression modeling to predict gene expression from histone modification ChIP-seq features. The model learns linear relationships between binned ChIP-seq signals and expression levels.

## Model Type
**Simple Linear Regression** (Ordinary Least Squares)
- No regularization
- No k fold cross-validation
- Learns interpretable linear coefficients for each feature
- Best for understanding direct feature-expression relationships

---

## Quick Start
```bash
# Train and evaluate the linear regression model
python models/scripts/LM.py
```

**Expected runtime:** ~2-5 minutes (depending on dataset size and number of bins)

---

## Workflow Overview

### Step 1: Data Loading and Splitting

**Input:** `models/data/combined_data.csv`

**Data Structure:**
- **Features (X):** All columns containing `_bin` (histone mark signals per bin)
- **Target (y):** `expression` column
- **Groups:** `gene_id` (ensures same gene doesn't appear in multiple splits)

**Splitting Strategy:**
```
Total Dataset
    ↓ GroupShuffleSplit (70/30)
    ├─ Train (70%)
    └─ Temp (30%)
           ↓ GroupShuffleSplit (50/50)
           ├─ Validation (15%)
           └─ Test (15%)
```

**Why GroupShuffleSplit?**
- Prevents **gene leakage**: Same gene's expression in different cell lines won't appear in both train and test
- Example: If ENSG00001 appears in training set, ALL cell line observations for ENSG00001 stay in training
- Ensures model generalizes to **unseen genes**, not just unseen samples

**Expected output:**
```
Split Results:
  Train: 53855 samples, 13464 genes
  Val:   11541 samples, 2885 genes
  Test:  11540 samples, 2885 genes
```

---

### Step 2: Model Training

**Model:** `LinearRegression()` (sklearn)
- No hyperparameters to tune
- Fits coefficients using ordinary least squares
- One coefficient per bin per histone mark

**Training:**
```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

**Model outputs:**
- **Coefficients:** Linear weights for each feature (bin)
- **Intercept:** Baseline expression level

---

### Step 3: Model Evaluation

**Metrics calculated:**
- **R² (R-squared):** Proportion of variance explained (0-1, higher is better)
- **RMSE (Root Mean Squared Error):** Prediction error magnitude (lower is better)
- **MAE (Mean Absolute Error):** Average absolute prediction error (lower is better)
- **Pearson r:** Linear correlation between predicted and true values (-1 to 1)

**Expected output structure:**
```
Simple Linear Regression Results:
  Train - R²: 0.8523, RMSE: 0.5642, MAE: 0.4201, Pearson: 0.9233
  Val   - R²: 0.7891, RMSE: 0.6534, MAE: 0.4876, Pearson: 0.8884
  Test  - R²: 0.7832, RMSE: 0.6621, MAE: 0.4912, Pearson: 0.8849

Overfitting gap (Train R² - Val R²): 0.0632
```

**Interpretation:**
- **R² > 0.75** on test set → Model explains >75% of expression variance
- **Small overfitting gap** (~0.06) → Model generalizes well
- **High Pearson r** → Strong linear relationship captured

---

### Step 4: Visualization

The script generates comprehensive visualizations automatically:

#### A. Prediction Plots
**File:** `analysis/figures/slr/linear_regression_predictions.png`

Shows predicted vs. true expression for train/val/test sets:
- **Diagonal line:** Perfect prediction
- **Points near line:** Good predictions
- **Scatter:** Prediction errors

#### B. Residual Analysis
**File:** `analysis/figures/slr/linear_regression_residuals.png`

**Top row:** Residuals vs Predicted (checks for heteroscedasticity)
- Points should be randomly scattered around y=0
- No funnel shape → homoscedastic errors ✓

**Bottom row:** Residual distribution (checks for normality)
- Should be approximately normal (bell curve)
- Mean ≈ 0 → unbiased predictions ✓

#### C. Learning Curves
**File:** `analysis/figures/slr/linear_regression_learning_curves.png`

Shows how performance changes with training set size:
- **Train score decreases** as data increases (less overfitting)
- **Val score increases** as data increases (better generalization)
- **Curves converge** → optimal sample size reached

---

### Step 5: Model Interpretation

The script provides extensive coefficient analysis to understand **which histone marks** and **where in gene regions** drive expression predictions.

#### A. Coefficient Heatmap
**File:** `analysis/figures/slr/lr_coefficient_heatmap_signed.png`

**Shows:** Coefficient value for each histone mark at each bin position
- **Red/Positive:** Activating effect (higher signal → higher expression)
- **Blue/Negative:** Repressive effect (higher signal → lower expression)
- **Yellow lines:** Gene body boundaries (5% upstream | gene body | 5% downstream)

**Key insights:**
- H3K4me3, H3K27ac → typically positive coefficients (activating marks)
- H3K27me3, H3K9me3 → typically negative coefficients (repressive marks)
- Strongest coefficients often at promoter/TSS region

#### B. Regional Importance Analysis
**Files:** 
- `analysis/figures/slr/lr_regional_importance_signed.png`
- `analysis/figures/slr/lr_regional_importance_absolute.png`

**Regions analyzed:**
1. **Upstream (5% gene length):** Promoter/regulatory region
2. **Gene Body (100% gene length):** Transcribed region
3. **Downstream (5% gene length):** 3' regulatory region

**Signed coefficients:** Shows direction of effect
- Positive → activating
- Negative → repressive

**Absolute coefficients:** Shows magnitude of effect
- Higher values → more important for prediction

#### C. Coefficient Profiles
**File:** `analysis/figures/slr/lr_coefficient_profiles.png`

**Shows:** How each histone mark's coefficient changes across gene regions
- X-axis: Bin position (upstream → gene body → downstream)
- Y-axis: Coefficient value
- Each line: One histone mark

**Biological insights:**
- H3K4me3: Peak at TSS (transcription start site)
- H3K27ac: Elevated in promoter and enhancer regions
- H3K36me3: Enriched in gene body
- H3K27me3/H3K9me3: Repressive throughout

#### D. Overall Histone Mark Importance
**Files:**
- `analysis/figures/slr/lr_histone_importance_signed.png`
- `analysis/figures/slr/lr_histone_importance_absolute.png`

**Shows:** Overall contribution of each histone mark (averaged across all bins)

**Typical ranking (by absolute importance):**
1. H3K4me3 (promoter mark, strongest activator)
2. H3K27ac (active enhancer/promoter)
3. H3K27me3 (Polycomb repression)
4. H3K4me1 (poised enhancer)
5. H3K9me3 (heterochromatin repression)

#### E. Comprehensive Summary
**File:** `analysis/figures/slr/lr_interpretation_summary_signed.png`

Multi-panel figure combining:
- Panel A: Coefficient heatmap
- Panel B: Regional importance (signed)
- Panel C: Regional importance (absolute)
- Panel D: Overall mark importance (signed)
- Panel E: Overall mark importance (absolute)
- Panel F: Coefficient profiles across gene regions

---

## Output Files

### Models
```
models/
├── trained/
│   └── linear_regression.pkl          # Trained model (can reload for predictions)
```

### Results
```
models/results/slr/
├── performance_metrics.csv            # R², RMSE, MAE, Pearson r for train/val/test
├── coefficient_values.csv             # All coefficients with feature names
├── coefficient_matrix.csv             # Pivot table: Histone_Mark × Bin
├── regional_importance_signed.csv     # Mean signed coef per region per mark
├── regional_importance_absolute.csv   # Mean absolute coef per region per mark
└── histone_mark_importance.csv        # Overall importance per mark
```

### Figures
```
analysis/figures/slr/
├── linear_regression_predictions.png              # Predicted vs true
├── linear_regression_residuals.png                # Residual analysis
├── linear_regression_learning_curves.png          # Sample size effects
├── lr_coefficient_heatmap_signed.png              # Heatmap
├── lr_coefficient_profiles.png                     # Profiles
├── lr_regional_importance_signed_and_absolute.png              # Regional
├── lr_histone_importance_signed_and_absolute.png               # Overall
└── lr_interpretation_summary_signed.png           # Comprehensive summary
```

---

## Key Parameters

### Data Splitting
```python
# In LM.py, line 24-25:
test_size=0.30          # 30% for validation + test
random_state=42         # For reproducibility
```

### Model Settings
```python
# Linear Regression (no hyperparameters)
lr_model = LinearRegression()
```

---

## Biological Interpretation Guide

### Understanding Coefficients

**Positive coefficient (+):**
- Higher ChIP-seq signal → Higher predicted expression
- Indicates **activating/permissive** chromatin state
- Expected for: H3K4me3, H3K27ac, H3K4me1, H3K36me3

**Negative coefficient (-):**
- Higher ChIP-seq signal → Lower predicted expression
- Indicates **repressive** chromatin state
- Expected for: H3K27me3, H3K9me3

**Magnitude:**
- Larger absolute value → Stronger effect on prediction
- Small values (~0.001) → Weak contribution
- Large values (~0.1+) → Strong contribution

### Regional Patterns

**Upstream (Promoter):**
- H3K4me3: Strong positive (marks active promoters)
- H3K27ac: Positive (marks active enhancers/promoters)
- H3K27me3: Negative (Polycomb-repressed genes)

**Gene Body:**
- H3K36me3: Positive (marks actively transcribed genes)
- H3K4me1: Variable (poised enhancers)

**Downstream:**
- Usually less important than promoter region
- Some marks show 3' enrichment patterns

---

## Troubleshooting

### Issue: Low R² on validation/test sets
**Solutions:**
- Check for data quality issues (outliers, missing values)
- Inspect residual plots for systematic errors
- Consider non-linear models (Random Forest, Neural Networks)
- Add more features or increase bin resolution

### Issue: Large overfitting gap (Train R² >> Val R²)
**Solutions:**
- Use Ridge regression instead (adds L2 regularization)
- Reduce number of features
- Get more training data

### Issue: Memory error during training
**Solutions:**
- Reduce `n_bins` in feature extraction (use 50 or 100 instead of 200)
- Use sparse matrix format
- Process in batches

### Issue: Residuals show patterns (not random)
**Solutions:**
- May indicate model is missing non-linear relationships
- Try polynomial features or non-linear models
- Check for batch effects in data

---

## Next Steps

### Model Improvements
1. **Ridge Regression:** Add L2 regularization to reduce overfitting
2. **Lasso Regression:** Add L1 regularization for feature selection
3. **ElasticNet:** Combine L1 and L2 regularization
4. **Random Forest:** Capture non-linear relationships
5. **Neural Network:** Learn complex feature interactions

### Feature Engineering
1. **Interaction terms:** Multiply features (e.g., H3K4me3 × H3K27ac)
2. **Ratios:** H3K27ac/H3K27me3 (active/repressive balance)
3. **Aggregations:** Mean signal over gene body vs. promoter
4. **Peak calling:** Binary features (peak present/absent)

### Advanced Analysis
1. **Gene-specific models:** Train separate models for different gene classes
2. **Cell-line effects:** Include cell line as categorical feature
3. **Cross-validation:** Use GroupKFold for robust evaluation
4. **Permutation importance:** Alternative feature importance method

---

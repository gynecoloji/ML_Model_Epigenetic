# Data Quality Checks for ML Modeling

## Overview

This script performs comprehensive pre-modeling data quality checks for genomic datasets, specifically designed for ChIP-seq histone modification data predicting gene expression levels. It systematically validates data integrity, distribution characteristics, feature properties, and correlation patterns before machine learning model training.

## Purpose

**Before training any ML model, you should:**
1. Verify data quality (missing values, duplicates, outliers)
2. Understand target variable distribution
3. Check feature characteristics (variance, scale, correlation)
4. Identify potential issues (multicollinearity, data leakage, imbalance)

This script automates these essential pre-modeling checks.

---

## Data Structure

### Expected Input Format

The script expects a CSV file with the following structure:

| gene_id | cell_line | expression | H3K4me3_bin1 | ... | H3K9me3_bin200 |
|---------|-----------|------------|--------------|-----|----------------|
| ENSG... | HeyA8     | 8.55       | 0.807        | ... | 0.530          |
| ENSG... | HeyA8     | 3.41       | 0.637        | ... | 0.579          |

**Columns:**
- `gene_id`: Gene identifier (e.g., ENSEMBL ID)
- `cell_line`: Cell line name (categorical)
- `expression`: Target variable (continuous)
- `{histone_mark}_bin{N}`: Histone modification features (N = 1-200)

**Features:**
- 5 histone marks: H3K4me3, H3K4me1, H3K9me3, H3K27me3, H3K27ac
- 200 bins per histone mark
- Total: 1000 features

---

## Usage

### Basic Usage

```python
import pandas as pd
from Data_glimpse import *

# Load your data
df = pd.read_csv("models/data/combined_data.csv")

# Run all quality checks
quality_results = check_data_quality(df)
target_results = analyze_target_variable(df)
feature_results = analyze_feature_characteristics(df)
correlation_results = analyze_feature_correlations(df)
```

### Custom Parameters

```python
# Customize analysis parameters
target_results = analyze_target_variable(
    df, 
    target_col='expression',
    group_col='cell_line',
    save_outputs=True  # Save figures and results
)

feature_results = analyze_feature_characteristics(
    df,
    variance_threshold=0.01,  # Threshold for low variance
    save_outputs=True
)

correlation_results = analyze_feature_correlations(
    df,
    high_corr_threshold=0.95,  # Threshold for multicollinearity
    top_n_features=20,         # Top N features to display
    save_outputs=True
)
```

---

## Functions

### 1. `check_data_quality()`

**Purpose:** Validates basic data integrity

**Checks:**
- Dataset dimensions (rows × columns)
- Data types distribution
- Missing values (count and percentage)
- Duplicate rows
- Duplicate gene-cell_line combinations
- Sample distribution across cell lines

**Returns:** Dictionary with quality metrics

**Output:**
```
==================================================
BASIC DATA QUALITY CHECKS
==================================================
Dataset shape: (72476, 1003)
  - 72,476 rows
  - 1,003 columns

Column types:
float64    1001
object        2

Missing values:
✓ No missing values!

Duplicate rows: 0
Duplicate gene_id+cell_line pairs: 0

Identifier summary:
  - Unique gene_id: 18,119
  - Unique cell_line: 4
```

---

### 2. `analyze_target_variable()`

**Purpose:** Characterizes the target variable (gene expression)

**Checks:**
- Descriptive statistics (mean, median, quartiles)
- **Distribution shape:**
  - **Skewness** → Symmetry of distribution
  - **Kurtosis** → Tail heaviness and outlier presence
- Outlier detection (IQR method)
- Distribution across cell lines

**Interpretations:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Skewness** | ~0 | Symmetric |
| | 0.5-1 | Moderate skew |
| | 1-2 | Strong skew |
| | >2 | Extremely skewed |
| **Excess Kurtosis** | <1 | Normal tails |
| | 1-3 | Heavy tails |
| | 3-5 | Moderately heavy tails |
| | ≥5 | Extreme outliers dominate |

**Visualizations:**
- Histogram with mean/median lines
- Q-Q plot (normality check)
- Boxplot by cell line

**Outputs:**
- Figure: `analysis/QC/target_distribution_expression.png`
- Text summary: `models/results/QC/target_analysis.txt`

---

### 3. `analyze_feature_characteristics()`

**Purpose:** Analyzes properties of all 1000 features efficiently

**Checks:**
- **Feature grouping** by histone mark
- **Variance analysis:**
  - Near-zero variance features (provide little information)
  - Variance distribution across features
- **Scale analysis:**
  - Feature ranges, means, standard deviations
  - Scale ratio (max range / min range)
  - Scaling recommendation
- **Missing values** per feature

**Key Outputs:**
```
Feature groups detected: ['H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K4me3', 'H3K9me3']
  - H3K27ac: 200 features
  - H3K27me3: 200 features
  ...

Near-zero variance features (var < 0.01):
  - Total: 15 (1.50%)

Scale analysis:
⚠ Large scale differences (ratio: 45.23)
  → Recommend standardization before modeling
```

**Outputs:**
- Text summary: `models/results/QC/feature_characteristics.txt`

---

### 4. `analyze_feature_correlations()`

**Purpose:** Identifies feature relationships and multicollinearity

**Checks:**

1. **Correlation with target:**
   - Top N most predictive features
   - Average correlation by histone mark
   
2. **Multicollinearity detection:**
   - Highly correlated feature pairs (|r| ≥ 0.95)
   - Patterns within/across histone marks
   
3. **Within-group correlation:**
   - Average correlation within each histone mark
   - Expected: Adjacent bins should be correlated (ChIP-seq property)

**Key Outputs:**
```
Top 20 features correlated with expression:
   1. H3K4me3_bin87              :  0.4532
   2. H3K27ac_bin92               :  0.4201
   ...

Highly correlated pairs (|r| >= 0.95):
  - Total pairs: 234
  
  Within same histone mark:
    198 pairs (84.6%)
```

**Outputs:**
- Text summary: `models/results/QC/correlation_analysis.txt`

---

## Output Directory Structure

After running the script, the following structure is created:

```
project_root/
├── analysis/
│   └── QC/
│       └── target_distribution_expression.png
│
└── models/
    └── results/
        └── QC/
            ├── target_analysis.txt
            ├── feature_characteristics.txt
            └── correlation_analysis.txt
```

**Directories:**
- `analysis/QC/` → Diagnostic figures
- `models/results/QC/` → Text summaries and detailed results

---

## Interpretation Guide

### What to Look For

#### ✅ **Good Signs**

| Check | Good Sign | Recommendation |
|-------|-----------|----------------|
| Missing values | 0% missing | Proceed |
| Duplicates | 0 duplicates | Proceed |
| Target skewness | \|skew\| < 1 | No transformation needed |
| Feature variance | Few near-zero variance features | Keep most features |
| Scale | Ratio < 10 | Scaling optional |
| Multicollinearity | Few pairs (|r| ≥ 0.95) | Low redundancy |

#### ⚠️ **Warning Signs**

| Check | Warning Sign | Action Required |
|-------|--------------|-----------------|
| Missing values | >5% missing in any feature | Impute or remove features |
| Duplicates | Duplicate gene-cell_line pairs | Remove duplicates |
| Target skewness | \|skew\| > 2 | Consider log transformation |
| Kurtosis | Excess kurtosis ≥ 5 | Check for extreme outliers |
| Feature variance | >10% near-zero variance | Remove low-variance features |
| Scale | Ratio > 10 | **Standardization required** |
| Multicollinearity | Many pairs (|r| ≥ 0.95) | Feature selection or PCA |

### Domain-Specific Expectations

**For ChIP-seq histone modification data:**
- Adjacent bins (e.g., bin85-bin86) **should be correlated** ✓
- High within-mark correlation is **normal** ✓
- Different histone marks may have different predictive power
- Expression distribution may be right-skewed (low expression common)

---

## Common ML Issues Detected

### 1. **Near-Zero Variance Features**
**Problem:** Features with little variation provide no predictive power  
**Solution:** Remove features with variance < threshold

### 2. **Scale Differences**
**Problem:** Features on different scales bias distance-based algorithms  
**Solution:** Standardization (z-score) or normalization (min-max)

### 3. **Multicollinearity**
**Problem:** Highly correlated features cause model instability  
**Solutions:**
- Remove one feature from each highly correlated pair
- PCA/dimensionality reduction
- Regularization (Ridge, Lasso)

### 4. **Outliers in Target**
**Problem:** Extreme values can dominate loss functions  
**Solutions:**
- Robust scaling
- Log transformation
- Robust regression methods

### 5. **Class Imbalance** (for classification)
**Problem:** Unequal representation of classes  
**Solutions:**
- Stratified sampling
- SMOTE/oversampling
- Class weights

---

## Next Steps After QC

Based on QC results, prepare your data:

```python
# 1. Remove low-variance features
low_var_features = feature_results['variance']['low_variance_features']
df_clean = df.drop(columns=low_var_features)

# 2. Handle highly correlated features
# (Keep only one from each highly correlated pair)

# 3. Scale features if needed
from sklearn.preprocessing import StandardScaler
if feature_results['scale']['needs_scaling']:
    scaler = StandardScaler()
    # Scale only feature columns, not IDs or target
    
# 4. Handle outliers if needed
# (Based on target analysis results)

# 5. Train/validation/test split
# (Stratify by cell_line to ensure balanced representation)
```

---

## References

### Statistical Concepts
- **Skewness:** Measures asymmetry of distribution
- **Kurtosis:** Measures tail heaviness and outlier proneness
- **IQR Method:** Q1 - 1.5×IQR, Q3 + 1.5×IQR for outlier detection
- **Pearson Correlation:** Linear relationship between variables

### ML Best Practices
- Always check data quality before modeling
- Understand your target variable distribution
- Address multicollinearity in high-dimensional data
- Scale features for distance-based algorithms
- Stratify train/test splits for grouped data

---

## Contact & Support

For questions or issues:
1. Check your data format matches expected structure
2. Review interpretation guide above
3. Examine saved QC reports in `models/results/QC/`

---

## License

MIT



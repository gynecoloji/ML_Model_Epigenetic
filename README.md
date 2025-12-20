# Chromatin Regulatory Modeling

## Overview
This project uses machine learning to predict gene expression levels from epigenetic histone modification ChIP-seq data. By leveraging histone marks as features, we aim to build predictive models that capture the regulatory relationships between chromatin states and transcriptional activity.

## Key Objectives
- Extract quantitative features from histone modification ChIP-seq data
- Build and train machine learning models to predict microarray/RNA expression levels
- Evaluate model performance and interpret feature importance
- Understand the relationship between epigenetic marks and gene regulation

## Directory Structure
```
ChIPseq_expression_prediction/
├── data/
│   └── expression/              # Public gene expression datasets (RNA-seq)
├── features/
│   ├── extracted/               # Processed feature files from ChIP-seq data
│   ├── scripts/                 # Scripts for feature extraction
│   └── sample_metadata         # Sample information and metadata files (not shown)
├── models/
│   ├── data/                    # Training/testing datasets for modeling
│   ├── scripts/                 # Model training, assessment, and interpretation scripts
│   ├── trained/                 # Saved trained models (.pkl, .h5, etc.)
│   └── results/                 # Model performance tables and metrics
├── analysis/
│   ├── notebooks/               # Jupyter notebooks for exploratory analysis
│   └── figures/                 # Model-related visualizations and plots
└── results/
    └── normalized_bigwig/       # Processed BigWig files from upstream workflows
```

### Folder Descriptions

#### `data/`
- **expression/**: Contains gene expression data from public repositories (e.g., GEO, ENCODE)
  - RNA-seq count matrices
  - Normalized expression values (TPM, FPKM)
  - microarray normalized data

#### `features/`
- **extracted/**: ChIP-seq derived features ready for modeling
  - Signal quantifications per gene/region
  - Aggregated histone modification profiles
- **scripts/**: Feature extraction pipeline scripts
- **sample_metadata.csv: Sample annotations, experimental conditions, batch information

#### `models/`
- **data/**: Prepared datasets for model training and testing
  - Train/validation/test splits
  - Scaled/transformed feature matrices
- **scripts/**: Machine learning pipeline scripts
  - Model training and hyperparameter tuning
  - Performance assessment and cross-validation
  - Feature importance and model interpretation
- **trained/**: Serialized trained models for deployment or further analysis
- **results/**: Model evaluation outputs
  - Performance metrics tables (R², RMSE, MAE)
  - Cross-validation results
  - Comparison tables across different models

#### `analysis/`
- **notebooks/**: Interactive analysis and prototyping
  - Exploratory data analysis (EDA)
  - Model development and testing
- **figures/**: Publication-quality visualizations
  - Feature importance plots
  - Prediction vs. actual scatter plots
  - Learning curves and model diagnostics

#### `results/`
- **normalized_bigwig/**: Output from upstream ChIP-seq processing workflows
  - Normalized coverage tracks
  - Quality-controlled BigWig files ready for feature extraction
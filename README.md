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
├── data/                        # reference files also included in this folder
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

## Data Sources

### ChIP-seq Data
- **GEO (Gene Expression Omnibus)**: Histone modification ChIP-seq datasets
  - H3K4me3, H3K27ac, H3K4me1, H3K27me3, H3K9me3
  - Download from: https://www.ncbi.nlm.nih.gov/Traces/study/?query_key=1&WebEnv=MCID_694730cd77b84e64e9fa6c62&o=acc_s%3Aa

### Expression Data
- **GEO (Gene Expression Omnibus)**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118171

### Genome Annotations
- Gene annotations (Gene_Models_Genecode_v42.gtf)
- Genomic reference (hg38)
- bed files generated from the gtf file
  - 5% gene body length upstream/downstream of gene and gene body
  - keep strand +/-
  - keep protein-coding genes on chr1-22 and chrX

## Usage & Workflow

### Step 0: Create Main Directory Tree
```python
python dir_tree_model.py
```
**Determination of the Model (simple linear model, random tree regression ...)**

----

### Step 1: Prepare Expression Data

**For Microarray Data:**
```R
# Run the microarray preparation script
Rscript prepare_microarray.R
```

This script will:
1. Download microarray data from GEO (e.g., GSE118171)
2. Extract expression values and phenotype data
3. Filter for samples of interest
4. Annotate probes with Ensembl gene IDs
5. Remove duplicates and invalid entries
6. Generate two output files:
   - `data/expression/gene_expression.csv` - Individual replicates
   - `data/expression/gene_expression_merged_mean.csv` - Mean across replicates

---

### Step 2: Extract Features from ChIP-seq Data
Extract quantitative features from histone modification ChIP-seq BigWig files at gene regions (promoters, gene bodies, etc.).

📁 **Detailed instructions:** `features/README.md`

**Input:** `results/normalized_bigwig/*.bw`  
**Output:** `features/extracted/chip_signals_per_gene.csv`

---

### Step 3: Prepare Modeling Dataset
Merge expression data with ChIP-seq features and create train/test splits.

📁 **Detailed instructions:** `models/<model_name>_README.md`

**Output:** 
- `models/data/combined_data.csv`

---

### Step 4: Train Models
Train machine learning models to predict expression from histone modification features.

📁 **Detailed instructions:** `models/<model_name>_README.md`

**Output:** `models/trained/<model_full_name>.pkl`

---

### Step 5: Evaluate Models
Assess model performance on held-out test data and generate performance metrics.

📁 **Detailed instructions:** `models/<model_name>_README.md`

**Output:** `models/results/<model_name>/*.csv`

---

### Step 6: Interpret Results
Analyze feature importance and generate visualizations to understand which histone marks drive predictions.

📁 **Detailed instructions:** `models/<model_name>_README.md`

**Output:** `analysis/figures/<model_name>/*.png`

---
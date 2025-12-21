# Features Extraction

## Overview
This folder contains scripts and data for extracting quantitative features from histone modification ChIP-seq data. Features are extracted as binned signals along gene bodies (5' to 3' orientation, strand-aware) and prepared for machine learning modeling.

## Directory Contents
```
features/
├── scripts/                    # Feature extraction scripts
│   ├── prepare_data.py        # Create sample metadata
│   ├── Sig_extract.py         # Extract ChIP signals per bin
│   └── Filter_chip_signal.py  # Filter and merge with expression
├── extracted/                  # Output: extracted features
│   └── chip_signals_per_gene.csv
└── sample_metadata.csv         # Sample-to-cell line mapping
```

---

## Workflow

### Step 1: Create Sample Metadata

**Script:** `prepare_data.py`

**Purpose:** Generate metadata mapping each ChIP-seq sample to its cell line and histone mark.

**Input:**
- `results/samples.csv` - Sample information from upstream ChIP-seq workflow

**Output:**
- `features/sample_metadata.csv` - Clean metadata table

**Columns in output:**
| Column | Description |
|--------|-------------|
| `sample_id` | Unique sample identifier |
| `cell_line` | Cell line (PEO1, OVCA429, SKOV3, HeyA8) |
| `histone_mark` | Histone modification (H3K4me3, H3K27ac, etc.) |
| `bigwig_path` | Path to normalized BigWig file |

**Run:**
```bash
python features/scripts/prepare_data.py
```

**Expected output:**
```
✓ Saved metadata to: features/sample_metadata.csv

Summary:
  - Total samples: 20
  - Cell lines: ['HeyA8', 'OVCA429', 'PEO1', 'SKOV3']
  - Histone marks: ['H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K4me3', 'H3K9me3']
```

---

### Step 2: Extract ChIP-seq Signals Per Bin

**Script:** `Sig_extract.py`

**Purpose:** Extract quantitative ChIP-seq signal values across gene bodies, divided into N bins (5' → 3' orientation, strand-aware).

**Input:**
- `data/genes_body_5per_chr1-22_X_protein_coding_width_gt_200.sorted.bed` - Gene body coordinates
- `features/sample_metadata.csv` - Sample metadata from Step 1
- `results/normalized_bigwig/*.bw` - Normalized BigWig files

**Output:**
- `features/extracted/chip_signals_per_gene.csv`

**Key Features:**
- **Strand-aware binning**: bin1 = 5' end, binN = 3' end (automatically reversed for minus-strand genes)
- **Parallel processing**: Uses multiple cores for faster extraction
- **Configurable bins**: Adjust `n_bins` parameter (default: 200)

**Parameters:**
```python
# In Sig_extract.py, adjust these parameters:
n_bins = 200        # Number of bins per gene (50, 100, 200, etc.)
n_jobs = 8          # Number of CPU cores to use (-1 = all cores)
```

**Run:**
```bash
python features/scripts/Sig_extract.py
```

**Expected output:**
```
📖 Reading gene body annotations...
  ✓ Loaded genes
  ✓ Strand distribution: {'+': 9,617, '-': 9,617}

📋 Reading sample metadata...
  ✓ Found 20 ChIP-seq samples

🔬 Extracting ChIP-seq signals (200 bins per gene, strand-aware)...
⚡ Using parallel processing with 8 cores...
[Processing samples...]

💾 Saving results to: features/extracted/chip_signals_per_gene.csv
```

**Output structure:**
```
gene_id    chr    start    end    strand    HeyA8_H3K4me3_bin1  ...  SKOV3_H3K9me3_bin200
ENSG001    chr1   1000     5000   +         2.34                ...  0.12
ENSG002    chr1   10000    15000  -         1.89                ...  0.45
...
```

---

### Step 3: Filter and Merge with Expression Data

**Script:** `Filter_chip_signal.py`

**Purpose:** 
1. Filter ChIP signal features (keep only bin columns)
2. Log-transform signals: log2(signal + 1)
3. Merge with expression data
4. Reshape to long format for modeling (gene-cellline pairs)

**Input:**
- `features/extracted/chip_signals_per_gene.csv` - ChIP signals from Step 2
- `data/expression/gene_expression_merged_mean.csv` - Expression data

**Output:**
- `models/data/combined_data.csv` - Final modeling dataset

**Data transformation:**

**Before (wide format):**
```
gene_id    HeyA8    OVCA429    PEO1    SKOV3    HeyA8_H3K4me3_bin1  ...
ENSG001    12.5     10.2       8.9     11.3     2.34                ...
ENSG002    8.7      9.1        7.5     8.2      1.89                ...
```

**After (long format):**
```
gene_id    cell_line    expression    H3K4me3_bin1    H3K27ac_bin1    ...
ENSG001    HeyA8        12.5          2.34            1.56            ...
ENSG001    OVCA429      10.2          2.01            1.32            ...
ENSG001    PEO1         8.9           1.87            1.21            ...
ENSG001    SKOV3        11.3          2.12            1.45            ...
ENSG002    HeyA8        8.7           1.89            0.98            ...
...
```

**Run:**
```bash
python features/scripts/Filter_chip_signal.py
```

**Expected output:**
```
  Saved to: models/data/combined_data.csv
```

---
## Output Data Structure

### `sample_metadata.csv`
```csv
sample_id,cell_line,histone_mark,bigwig_path
sample_001,HeyA8,H3K4me3,results/normalized_bigwig/sample_001.normalized.bw
sample_002,HeyA8,H3K27ac,results/normalized_bigwig/sample_002.normalized.bw
...
```

### `chip_signals_per_gene.csv`
- **Rows:** Genes (one row per gene)
- **Columns:** 
  - Metadata: `gene_id`, `chr`, `start`, `end`, `strand`
  - Features: `{CellLine}_{HistoneMark}_bin{N}` (e.g., `HeyA8_H3K4me3_bin1`)

### `combined_data.csv` (final modeling dataset)
- **Rows:** Gene-cellline pairs (genes × cell lines)
- **Columns:**
  - `gene_id`: Ensembl gene ID
  - `cell_line`: Cell line identifier
  - `expression`: Gene expression value (target variable)
  - `{HistoneMark}_bin{N}`: Log2-transformed ChIP signal features

---

## Key Parameters

### Number of Bins (`n_bins`)
Controls the resolution of feature extraction:
- **50 bins**: Coarse resolution, faster processing
- **100 bins**: Medium resolution
- **200 bins**: Fine resolution (default), captures more spatial detail
- **Higher**: More features but risk of overfitting

**To change:** Edit `n_bins` parameter in `Sig_extract.py`

### Parallel Processing (`n_jobs`)
Controls CPU usage:
- **-1**: Use all available cores (fastest)
- **1**: Single core (slowest, but stable)
- **N**: Use N cores

**To change:** Edit `n_jobs` parameter in `Sig_extract.py`

---

## Notes

- **Strand orientation:** All bins are oriented 5' → 3', meaning bin1 always represents the transcription start site region
- **Missing values:** Genes with no ChIP signal are assigned 0
- **Log transformation:** Applied to reduce skewness: `log2(signal + 1)`
- **Gene filtering:** Only protein-coding genes on chr1-22 and chrX with width > 200bp
# ============================================================================
# STEP-BY-STEP DEEP LEARNING PIPELINE
# Gene Expression Prediction from ChIP-seq Histone Modifications
# ============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

os.chdir("ML")
# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"{'='*70}\n")

# Create output directories
os.makedirs('models/trained/hybrid_DL', exist_ok=True)
os.makedirs('analysis/figures/hybrid_DL', exist_ok=True)
os.makedirs('models/results/hybrid_DL', exist_ok=True)

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================

# Load your dataframe
# df = pd.read_csv('your_data.csv')  # Uncomment and adjust path
# OR if you already have df loaded, skip this

df = pd.read_csv("models/data/combined_data.csv")
print("="*70)
print("STEP 1: DATA INSPECTION")
print("="*70)

# Inspect the dataframe
print(f"\nDataframe shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")

# Check for required columns
print(f"\nGene IDs: {df['gene_id'].nunique():,} unique genes")
print(f"Cell lines: {df['cell_line'].unique()}")
print(f"Expression range: [{df['expression'].min():.2f}, {df['expression'].max():.2f}]")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Show first few rows
print(f"\nFirst 3 rows:")
print(df.head(3))

print("\n" + "="*70)
print("✓ Data loaded successfully!")
print("="*70 + "\n")

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================

def prepare_pytorch_data(df, normalize=True):
    """
    Prepare data for PyTorch models.
    Converts tabular data to 3D format: (samples, channels, bins)
    """
    print("="*70)
    print("STEP 2: DATA PREPARATION")
    print("="*70)
    
    # Define parameters
    histone_marks = ['H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K27me3', 'H3K9me3']
    n_bins = 200
    n_marks = len(histone_marks)
    n_samples = len(df)
    
    # Extract target and metadata
    y = df['expression'].values.astype(np.float32)
    genes = df['gene_id'].values
    cell_lines = df['cell_line'].values
    
    print(f"\nExtracting features for {n_marks} histone marks...")
    
    # Initialize 3D array: (samples, channels, bins)
    # PyTorch format: channels first!
    X = np.zeros((n_samples, n_marks, n_bins), dtype=np.float32)
    
    # Fill array and normalize each mark separately
    scalers = {}
    for mark_idx, mark in enumerate(histone_marks):
        print(f"  Processing {mark}...", end=" ")
        
        # Get column names for this mark
        mark_cols = [f"{mark}_bin{i}" for i in range(1, n_bins + 1)]
        
        # Extract data
        mark_data = df[mark_cols].values.astype(np.float32)
        
        if normalize:
            # Normalize each mark independently
            scaler = StandardScaler()
            mark_data = scaler.fit_transform(mark_data)
            scalers[mark] = scaler
        
        # Assign to channel
        X[:, mark_idx, :] = mark_data
        
        print(f"✓ Shape: {mark_data.shape}, Range: [{mark_data.min():.2f}, {mark_data.max():.2f}]")
    
    print(f"\n{'='*70}")
    print("DATA PREPARATION SUMMARY")
    print(f"{'='*70}")
    print(f"Input shape:  {X.shape} (samples, channels, bins)")
    print(f"Output shape: {y.shape}")
    print(f"Data type:    {X.dtype}")
    print(f"Normalized:   {normalize}")
    print(f"\nUnique genes:      {len(np.unique(genes)):,}")
    print(f"Cell lines:        {np.unique(cell_lines)}")
    print(f"Expression range:  [{y.min():.2f}, {y.max():.2f}]")
    print(f"Expression mean:   {y.mean():.2f}")
    print(f"Expression std:    {y.std():.2f}")
    print("="*70 + "\n")
    
    return X, y, genes, cell_lines, scalers if normalize else None


# Run data preparation
X, y, genes, cell_lines, scalers = prepare_pytorch_data(df, normalize=True)

# Quick verification
print("Verification:")
print(f"  X contains NaN? {np.isnan(X).any()}")
print(f"  y contains NaN? {np.isnan(y).any()}")
print(f"  First sample X shape: {X[0].shape}")
print(f"  First sample y value: {y[0]:.4f}")

print("\n✓ STEP 2 COMPLETED - Data is prepared for PyTorch!")
print("Ready to proceed to Step 3 (Gene-based Splitting)?\n")

# ============================================================================
# STEP 3: GENE-BASED SPLITTING
# ============================================================================

def gene_based_split(X, y, genes, cell_lines, 
                     test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data ensuring all observations of the same gene stay together.
    CRITICAL: Prevents data leakage!
    """
    print("="*70)
    print("STEP 3: GENE-BASED SPLITTING")
    print("="*70)
    print("\nSplitting strategy:")
    print(f"  Test size:  {test_size*100:.0f}% of unique genes")
    print(f"  Val size:   {val_size*100:.0f}% of unique genes")
    print(f"  Train size: {(1-test_size-val_size)*100:.0f}% of unique genes")
    
    # First split: separate test set
    print(f"\nStep 3.1: Separating test set...")
    splitter_test = GroupShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=random_state
    )
    train_val_idx, test_idx = next(splitter_test.split(X, y, groups=genes))
    
    print(f"  ✓ Test set separated: {len(test_idx):,} samples")
    
    # Second split: separate validation from training
    print(f"\nStep 3.2: Separating validation set from remaining data...")
    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    genes_train_val = genes[train_val_idx]
    
    # Adjust val_size to be relative to remaining data
    adjusted_val_size = val_size / (1 - test_size)
    
    splitter_val = GroupShuffleSplit(
        n_splits=1, 
        test_size=adjusted_val_size, 
        random_state=random_state
    )
    train_idx_rel, val_idx_rel = next(splitter_val.split(
        X_train_val, y_train_val, groups=genes_train_val
    ))
    
    # Get actual indices
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    
    print(f"  ✓ Validation set separated: {len(val_idx):,} samples")
    print(f"  ✓ Training set remaining: {len(train_idx):,} samples")
    
    # Create splits
    splits = {
        'X_train': X[train_idx], 
        'y_train': y[train_idx],
        'genes_train': genes[train_idx], 
        'cell_lines_train': cell_lines[train_idx],
        
        'X_val': X[val_idx], 
        'y_val': y[val_idx],
        'genes_val': genes[val_idx], 
        'cell_lines_val': cell_lines[val_idx],
        
        'X_test': X[test_idx], 
        'y_test': y[test_idx],
        'genes_test': genes[test_idx], 
        'cell_lines_test': cell_lines[test_idx],
        
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    
    # Verify no gene overlap (CRITICAL CHECK!)
    print(f"\nStep 3.3: Verifying no gene overlap...")
    genes_train_set = set(splits['genes_train'])
    genes_val_set = set(splits['genes_val'])
    genes_test_set = set(splits['genes_test'])
    
    overlap_train_val = genes_train_set & genes_val_set
    overlap_train_test = genes_train_set & genes_test_set
    overlap_val_test = genes_val_set & genes_test_set
    
    assert len(overlap_train_val) == 0, f"ERROR: {len(overlap_train_val)} genes overlap between train and val!"
    assert len(overlap_train_test) == 0, f"ERROR: {len(overlap_train_test)} genes overlap between train and test!"
    assert len(overlap_val_test) == 0, f"ERROR: {len(overlap_val_test)} genes overlap between val and test!"
    
    print(f"  ✓ NO GENE OVERLAP detected - splits are valid!")
    
    # Print detailed statistics
    print(f"\n{'='*70}")
    print("SPLIT SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nDataset Sizes:")
    print(f"  {'Split':<10} {'Samples':<10} {'Unique Genes':<15} {'Percentage':<12}")
    print(f"  {'-'*50}")
    total_samples = len(X)
    total_genes = len(np.unique(genes))
    
    for split_name, split_genes in [('Train', splits['genes_train']), 
                                      ('Val', splits['genes_val']), 
                                      ('Test', splits['genes_test'])]:
        n_samples = len(split_genes)
        n_genes = len(np.unique(split_genes))
        pct = (n_samples / total_samples) * 100
        print(f"  {split_name:<10} {n_samples:<10,} {n_genes:<15,} {pct:<12.1f}%")
    
    print(f"\nCell Line Distribution:")
    for split_name, split_cell_lines in [('Train', splits['cell_lines_train']), 
                                          ('Val', splits['cell_lines_val']), 
                                          ('Test', splits['cell_lines_test'])]:
        unique, counts = np.unique(split_cell_lines, return_counts=True)
        print(f"\n  {split_name}:")
        for cell, count in zip(unique, counts):
            pct = (count / len(split_cell_lines)) * 100
            print(f"    {cell:<10}: {count:>6,} samples ({pct:>5.1f}%)")
    
    print(f"\nExpression Statistics:")
    print(f"  {'Split':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*50}")
    for split_name, split_y in [('Train', splits['y_train']), 
                                 ('Val', splits['y_val']), 
                                 ('Test', splits['y_test'])]:
        print(f"  {split_name:<10} {split_y.mean():<10.2f} {split_y.std():<10.2f} "
              f"{split_y.min():<10.2f} {split_y.max():<10.2f}")
    
    print("="*70 + "\n")
    
    return splits


# Run gene-based splitting
splits = gene_based_split(X, y, genes, cell_lines, 
                          test_size=0.15, val_size=0.15, random_state=42)

# Additional verification
print("Additional Verification:")
print(f"  Train + Val + Test samples = {len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']):,}")
print(f"  Original total samples     = {len(X):,}")
print(f"  Match? {len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']) == len(X)}")

# Check a specific gene to verify it's only in one split
sample_gene = genes[0]
in_train = sample_gene in splits['genes_train']
in_val = sample_gene in splits['genes_val']
in_test = sample_gene in splits['genes_test']
print(f"\nExample gene '{sample_gene}' appears in:")
print(f"  Train: {in_train}, Val: {in_val}, Test: {in_test}")
print(f"  ✓ Only in one split: {sum([in_train, in_val, in_test]) == 1}")

print("\n✓ STEP 3 COMPLETED - Data is properly split!")
print("Ready to proceed to Step 4 (Create DataLoaders)?\n")

# ============================================================================
# STEP 4: PYTORCH DATASET & DATALOADER
# ============================================================================

class ChIPSeqDataset(Dataset):
    """
    PyTorch Dataset for ChIP-seq histone modification data.
    """
    
    def __init__(self, X, y, genes=None, cell_lines=None):
        """
        Parameters:
        -----------
        X : numpy array (n_samples, n_channels, n_bins)
        y : numpy array (n_samples,)
        genes : optional gene IDs
        cell_lines : optional cell line names
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.genes = genes
        self.cell_lines = cell_lines
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """Returns a single sample."""
        return self.X[idx], self.y[idx]
    
    def get_metadata(self, idx):
        """Get gene and cell line info for a sample."""
        if self.genes is not None and self.cell_lines is not None:
            return self.genes[idx], self.cell_lines[idx]
        return None, None


def create_dataloaders(splits, batch_size=64, num_workers=4, pin_memory=True):
    """
    Create DataLoaders for train, validation, and test sets.
    """
    print("="*70)
    print("STEP 4: CREATING DATALOADERS")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Batch size:   {batch_size}")
    print(f"  Num workers:  {num_workers} (parallel data loading)")
    print(f"  Pin memory:   {pin_memory} (faster GPU transfer)")
    
    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = ChIPSeqDataset(
        splits['X_train'], splits['y_train'], 
        splits['genes_train'], splits['cell_lines_train']
    )
    print(f"  ✓ Train dataset: {len(train_dataset):,} samples")
    
    val_dataset = ChIPSeqDataset(
        splits['X_val'], splits['y_val'],
        splits['genes_val'], splits['cell_lines_val']
    )
    print(f"  ✓ Val dataset:   {len(val_dataset):,} samples")
    
    test_dataset = ChIPSeqDataset(
        splits['X_test'], splits['y_test'],
        splits['genes_test'], splits['cell_lines_test']
    )
    print(f"  ✓ Test dataset:  {len(test_dataset):,} samples")
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    print(f"  ✓ Train loader: {len(train_loader)} batches (shuffled)")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    print(f"  ✓ Val loader:   {len(val_loader)} batches")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for test
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    print(f"  ✓ Test loader:  {len(test_loader)} batches")
    
    print(f"\n{'='*70}")
    print("DATALOADER SUMMARY")
    print(f"{'='*70}")
    print(f"  Total batches: {len(train_loader) + len(val_loader) + len(test_loader)}")
    print(f"  Estimated batches per epoch: {len(train_loader)}")
    print(f"  Last batch size (train): ~{len(train_dataset) % batch_size if len(train_dataset) % batch_size != 0 else batch_size}")
    print("="*70 + "\n")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def verify_dataloader(loader, dataset_name, device, n_batches=2):
    """
    Verify dataloader works and check data shapes.
    """
    print(f"\nVerifying {dataset_name} DataLoader:")
    print(f"  Loading {n_batches} batches for testing...")
    
    for i, (batch_X, batch_y) in enumerate(loader):
        if i >= n_batches:
            break
        
        # Move to device
        batch_X_gpu = batch_X.to(device)
        batch_y_gpu = batch_y.to(device)
        
        print(f"\n  Batch {i+1}:")
        print(f"    X shape:  {batch_X.shape}")
        print(f"    y shape:  {batch_y.shape}")
        print(f"    X device: {batch_X_gpu.device}")
        print(f"    y device: {batch_y_gpu.device}")
        print(f"    X range:  [{batch_X.min():.3f}, {batch_X.max():.3f}]")
        print(f"    y range:  [{batch_y.min():.3f}, {batch_y.max():.3f}]")
    
    print(f"\n  ✓ {dataset_name} DataLoader working correctly!")


# Create dataloaders
loaders = create_dataloaders(
    splits, 
    batch_size=64,      # Adjust based on your GPU memory
    num_workers=4,      # Adjust based on your CPU cores (0-8 typically)
    pin_memory=True     # Set to False if you have limited RAM
)

# Verify dataloaders work correctly
print("\n" + "="*70)
print("VERIFICATION: Testing DataLoaders")
print("="*70)

verify_dataloader(loaders['train_loader'], 'Train', device, n_batches=2)
verify_dataloader(loaders['val_loader'], 'Validation', device, n_batches=1)
verify_dataloader(loaders['test_loader'], 'Test', device, n_batches=1)

print("\n" + "="*70)
print("✓ STEP 4 COMPLETED - DataLoaders are ready!")
print("="*70)
print("\nNext: We'll build the model architectures (CNN+Attention and CNN+LSTM)")
print("Ready to proceed to Step 5 (Build Models)?\n")

# ============================================================================
# STEP 5: MODEL ARCHITECTURES
# ============================================================================

print("="*70)
print("STEP 5: BUILDING MODEL ARCHITECTURES")
print("="*70)

# ============================================================================
# MODEL 1: CNN + ATTENTION
# ============================================================================

class CNNAttentionModel(nn.Module):
    """
    Hybrid CNN + Attention model for gene expression prediction.
    
    Architecture:
    1. Multi-channel 1D CNN - extracts local patterns from histone marks
    2. Self-attention - identifies important genomic regions
    3. Fully connected layers - final prediction
    """
    
    def __init__(self, n_channels=5, n_bins=200, 
                 conv_filters=[64, 128, 256], kernel_sizes=[7, 5, 3],
                 attention_heads=4, dropout=0.3):
        super(CNNAttentionModel, self).__init__()
        
        print("\nBuilding CNN+Attention Model...")
        
        # ====== CONVOLUTIONAL LAYERS ======
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = n_channels
        current_length = n_bins
        
        print("  Convolutional layers:")
        for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2)
            )
            self.batch_norms.append(nn.BatchNorm1d(filters))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
            print(f"    Layer {i+1}: Conv1d({in_channels} → {filters}, kernel={kernel_size}) + BatchNorm + MaxPool")
            
            in_channels = filters
            current_length = current_length // 2
        
        self.final_conv_channels = conv_filters[-1]
        self.final_length = current_length
        print(f"    Output: ({self.final_conv_channels} channels, {self.final_length} length)")
        
        # ====== ATTENTION MECHANISM ======
        print(f"\n  Attention layer:")
        self.attention = nn.MultiheadAttention(
            embed_dim=self.final_conv_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=False
        )
        self.attention_norm = nn.LayerNorm(self.final_conv_channels)
        print(f"    MultiheadAttention({attention_heads} heads, embed_dim={self.final_conv_channels})")
        
        # ====== FULLY CONNECTED LAYERS ======
        self.dropout = nn.Dropout(dropout)
        flattened_size = self.final_conv_channels * self.final_length
        
        print(f"\n  Fully connected layers:")
        print(f"    Input size: {flattened_size}")
        print(f"    FC1: {flattened_size} → 512")
        print(f"    FC2: 512 → 128")
        print(f"    FC3: 128 → 1 (output)")
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # CNN feature extraction
        for conv, bn, pool in zip(self.conv_layers, self.batch_norms, self.pools):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            x = pool(x)
        
        # Attention mechanism
        batch_size = x.shape[0]
        x = x.permute(2, 0, 1)  # (length, batch, channels)
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.attention_norm(x + attn_output)  # Residual connection
        x = x.permute(1, 2, 0)  # back to (batch, channels, length)
        
        # Flatten and predict
        x = x.reshape(batch_size, -1)
        x = self.dropout(x)
        output = self.fc_layers(x)
        
        return output.squeeze()


# ============================================================================
# MODEL 2: CNN + LSTM
# ============================================================================

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN + LSTM model for gene expression prediction.
    
    Architecture:
    1. Multi-channel 1D CNN - extracts local patterns
    2. Bidirectional LSTM - captures long-range dependencies
    3. Fully connected layers - final prediction
    """
    
    def __init__(self, n_channels=5, n_bins=200,
                 conv_filters=[64, 128, 256], kernel_sizes=[7, 5, 3],
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        
        print("\nBuilding CNN+LSTM Model...")
        
        # ====== CONVOLUTIONAL LAYERS ======
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = n_channels
        current_length = n_bins
        
        print("  Convolutional layers:")
        for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2)
            )
            self.batch_norms.append(nn.BatchNorm1d(filters))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
            print(f"    Layer {i+1}: Conv1d({in_channels} → {filters}, kernel={kernel_size}) + BatchNorm + MaxPool")
            
            in_channels = filters
            current_length = current_length // 2
        
        self.final_conv_channels = conv_filters[-1]
        self.final_length = current_length
        print(f"    Output: ({self.final_conv_channels} channels, {self.final_length} length)")
        
        # ====== LSTM LAYERS ======
        print(f"\n  LSTM layers:")
        self.lstm = nn.LSTM(
            input_size=self.final_conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        lstm_output_size = lstm_hidden * 2  # Bidirectional doubles the size
        print(f"    Bidirectional LSTM({lstm_layers} layers, hidden={lstm_hidden})")
        print(f"    Output size: {lstm_output_size} (bidirectional)")
        
        # ====== FULLY CONNECTED LAYERS ======
        self.dropout = nn.Dropout(dropout)
        
        print(f"\n  Fully connected layers:")
        print(f"    Input size: {lstm_output_size}")
        print(f"    FC1: {lstm_output_size} → 256")
        print(f"    FC2: 256 → 128")
        print(f"    FC3: 128 → 1 (output)")
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # CNN feature extraction
        for conv, bn, pool in zip(self.conv_layers, self.batch_norms, self.pools):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            x = pool(x)
        
        # LSTM processing
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)  # (batch, length, channels)
        lstm_out, (hidden, cell) = self.lstm(x)
        x = lstm_out[:, -1, :]  # Use last timestep output
        
        # Predict
        x = self.dropout(x)
        output = self.fc_layers(x)
        
        return output.squeeze()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_forward(model, model_name, device, input_shape=(4, 5, 200)):
    """
    Test forward pass with dummy data.
    """
    print(f"\nTesting {model_name} forward pass:")
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    print(f"  Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[:3].cpu().numpy()}")
    print(f"  ✓ Forward pass successful!")
    
    model.train()


# ============================================================================
# CREATE MODELS
# ============================================================================

print("\n" + "="*70)
print("CREATING MODELS")
print("="*70)

# Create CNN+Attention model
cnn_attn_model = CNNAttentionModel(
    n_channels=5,
    n_bins=200,
    conv_filters=[64, 128, 256],
    kernel_sizes=[7, 5, 3],
    attention_heads=4,
    dropout=0.3
).to(device)

print(f"\n  ✓ CNN+Attention created")
print(f"    Total parameters: {count_parameters(cnn_attn_model):,}")
print(f"    Model size: ~{count_parameters(cnn_attn_model) * 4 / (1024**2):.2f} MB")

# Create CNN+LSTM model
cnn_lstm_model = CNNLSTMModel(
    n_channels=5,
    n_bins=200,
    conv_filters=[64, 128, 256],
    kernel_sizes=[7, 5, 3],
    lstm_hidden=128,
    lstm_layers=2,
    dropout=0.3
).to(device)

print(f"\n  ✓ CNN+LSTM created")
print(f"    Total parameters: {count_parameters(cnn_lstm_model):,}")
print(f"    Model size: ~{count_parameters(cnn_lstm_model) * 4 / (1024**2):.2f} MB")

# Test forward passes
print("\n" + "="*70)
print("TESTING MODELS")
print("="*70)

test_model_forward(cnn_attn_model, "CNN+Attention", device)
test_model_forward(cnn_lstm_model, "CNN+LSTM", device)

print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)
print(f"\nBoth models successfully created and moved to {device}")
print(f"\nCNN+Attention: {count_parameters(cnn_attn_model):,} parameters")
print(f"CNN+LSTM:      {count_parameters(cnn_lstm_model):,} parameters")

print("\n" + "="*70)
print("✓ STEP 5 COMPLETED - Models are ready for training!")
print("="*70)
print("\nNext: We'll set up the training loop with:")
print("  - Loss function (MSE)")
print("  - Optimizer (Adam)")
print("  - Learning rate scheduler")
print("  - Early stopping")
print("  - Model checkpointing")
print("\nReady to proceed to Step 6 (Training Setup)?\n")

# ============================================================================
# STEP 6: TRAINING SETUP & LOOP
# ============================================================================

print("="*70)
print("STEP 6: TRAINING SETUP")
print("="*70)

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience=15, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.verbose:
                print(f"    → Initial best val loss: {val_loss:.4f}")
        elif val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"    → Val loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"    → No improvement ({self.counter}/{self.patience} epochs)")
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson': pearsonr(y_true, y_pred)[0],
        'spearman': spearmanr(y_true, y_pred)[0]
    }
    return metrics


def train_epoch(model, loader, criterion, optimizer, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_X, batch_y in loader:
        # Move to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    return avg_loss


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Store predictions and targets
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / n_batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return avg_loss, all_preds, all_targets


def train_model(model, train_loader, val_loader, model_name,
                n_epochs=100, learning_rate=0.001, weight_decay=1e-5,
                patience=15, clip_grad=1.0, scheduler_patience=5,
                device=device, verbose=True):
    """
    Complete training loop with all features.
    """
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Device:           {device}")
    print(f"  Parameters:       {count_parameters(model):,}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Weight decay:     {weight_decay}")
    print(f"  Max epochs:       {n_epochs}")
    print(f"  Patience:         {patience}")
    print(f"  Gradient clip:    {clip_grad}")
    print(f"  Scheduler patience: {scheduler_patience}")
    print(f"{'='*70}\n")
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Optimizer (Adam with weight decay for L2 regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (FIXED - removed verbose parameter)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=scheduler_patience,
        min_lr=1e-7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    
    # Training history
    history = defaultdict(list)
    
    # Best model tracking
    best_val_loss = float('inf')
    best_epoch = 0
    checkpoint_path = f"models/trained/hybrid_DL/{model_name}_best.pt"
    
    # Start training
    start_time = time.time()
    
    print("Starting training loop...")
    print(f"{'='*70}\n")
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, 
                                 optimizer, device, clip_grad)
        
        # Validate
        val_loss, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(val_targets, val_preds)
        
        # Store current learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        old_lr = current_lr
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manually print LR change if it happened
        if new_lr != old_lr and verbose:
            print(f"  → Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Store history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(new_lr)
        for key, value in val_metrics.items():
            history[f'val_{key}'].append(value)
        
        # Print progress (every 5 epochs or first/last)
        epoch_time = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val R²:     {val_metrics['r2']:.4f}")
            print(f"  Val Pearson: {val_metrics['pearson']:.4f}")
            print(f"  LR:         {new_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'metrics': val_metrics,
                'history': dict(history)
            }, checkpoint_path)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"    → Model saved! (val_loss={val_loss:.4f})")
        
        # Check early stopping
        if early_stopping(val_loss, epoch):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            print(f"  Best epoch was: {early_stopping.best_epoch}")
            print(f"  Best val loss: {early_stopping.best_loss:.4f}")
            break
        
        if epoch % 5 == 0 or epoch == 1:
            print()
    
    # Training summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time:     {total_time/60:.2f} minutes")
    print(f"Epochs trained: {epoch}")
    print(f"Best epoch:     {best_epoch}")
    print(f"Best val loss:  {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    return history, checkpoint_path



# ============================================================================
# QUICK TEST OF TRAINING COMPONENTS
# ============================================================================

print("\nTesting training components...")

# Test one training epoch
print("\n  Testing train_epoch function:")
criterion_test = nn.MSELoss()
optimizer_test = optim.Adam(cnn_attn_model.parameters(), lr=0.001)

sample_train_loss = train_epoch(
    cnn_attn_model, 
    loaders['train_loader'], 
    criterion_test, 
    optimizer_test, 
    device, 
    clip_grad=1.0
)
print(f"    ✓ Sample train loss: {sample_train_loss:.4f}")

# Test one validation epoch
print("\n  Testing validate_epoch function:")
sample_val_loss, sample_preds, sample_targets = validate_epoch(
    cnn_attn_model,
    loaders['val_loader'],
    criterion_test,
    device
)
print(f"    ✓ Sample val loss: {sample_val_loss:.4f}")
print(f"    ✓ Predictions shape: {sample_preds.shape}")
print(f"    ✓ Targets shape: {sample_targets.shape}")

# Test metrics calculation
sample_metrics = calculate_metrics(sample_targets, sample_preds)
print(f"\n  Testing calculate_metrics function:")
for k, v in sample_metrics.items():
    print(f"    {k}: {v:.4f}")

print("\n" + "="*70)
print("✓ STEP 6 COMPLETED - Training components ready!")
print("="*70)

print("\n" + "="*70)
print("READY TO TRAIN MODELS!")
print("="*70)
print("\nYou can now train the models using:")
print("\n  # Train CNN+Attention model")
print("  history_attn, checkpoint_attn = train_model(")
print("      model=cnn_attn_model,")
print("      train_loader=loaders['train_loader'],")
print("      val_loader=loaders['val_loader'],")
print("      model_name='CNNAttention',")
print("      n_epochs=100,")
print("      learning_rate=0.001,")
print("      weight_decay=1e-5,")
print("      patience=15")
print("  )")
print("\n  # Train CNN+LSTM model")
print("  history_lstm, checkpoint_lstm = train_model(")
print("      model=cnn_lstm_model,")
print("      train_loader=loaders['train_loader'],")
print("      val_loader=loaders['val_loader'],")
print("      model_name='CNNLSTM',")
print("      n_epochs=100,")
print("      learning_rate=0.001,")
print("      weight_decay=1e-5,")
print("      patience=15")
print("  )")
print("\nReady to train? Type 'yes' to start training both models!\n")

# ============================================================================
# TRAINING BOTH MODELS
# ============================================================================

print("\n" + "#"*70)
print("# STARTING TRAINING FOR BOTH MODELS")
print("#"*70)
print("\nThis may take 10-30 minutes depending on your GPU...")
print("Training will stop early if validation loss stops improving.\n")

# ============================================================================
# TRAIN MODEL 1: CNN + ATTENTION
# ============================================================================

print("\n" + "="*70)
print("MODEL 1/2: CNN + ATTENTION")
print("="*70)

history_attn, checkpoint_attn = train_model(
    model=cnn_attn_model,
    train_loader=loaders['train_loader'],
    val_loader=loaders['val_loader'],
    model_name='CNNAttention',
    n_epochs=100,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=15,
    clip_grad=1.0,
    scheduler_patience=5,
    device=device,
    verbose=True
)

print("\n✓ CNN+Attention training completed!")
print(f"  Best model saved at: {checkpoint_attn}")

# Quick plot of training history
print("\nPlotting CNN+Attention training history...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss
axes[0].plot(history_attn['epoch'], history_attn['train_loss'], label='Train', linewidth=2)
axes[0].plot(history_attn['epoch'], history_attn['val_loss'], label='Val', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('CNN+Attention: Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# R²
axes[1].plot(history_attn['epoch'], history_attn['val_r2'], color='green', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R²')
axes[1].set_title('CNN+Attention: Validation R²')
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(history_attn['epoch'], history_attn['learning_rate'], color='orange', linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('CNN+Attention: Learning Rate')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/hybrid_DL/CNNAttention_training_progress.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training plot saved to: analysis/figures/hybrid_DL/CNNAttention_training_progress.png\n")

# ============================================================================
# TRAIN MODEL 2: CNN + LSTM
# ============================================================================

print("\n" + "="*70)
print("MODEL 2/2: CNN + LSTM")
print("="*70)

history_lstm, checkpoint_lstm = train_model(
    model=cnn_lstm_model,
    train_loader=loaders['train_loader'],
    val_loader=loaders['val_loader'],
    model_name='CNNLSTM',
    n_epochs=100,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=15,
    clip_grad=1.0,
    scheduler_patience=5,
    device=device,
    verbose=True
)

print("\n✓ CNN+LSTM training completed!")
print(f"  Best model saved at: {checkpoint_lstm}")

# Quick plot of training history
print("\nPlotting CNN+LSTM training history...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss
axes[0].plot(history_lstm['epoch'], history_lstm['train_loss'], label='Train', linewidth=2)
axes[0].plot(history_lstm['epoch'], history_lstm['val_loss'], label='Val', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('CNN+LSTM: Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# R²
axes[1].plot(history_lstm['epoch'], history_lstm['val_r2'], color='green', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R²')
axes[1].set_title('CNN+LSTM: Validation R²')
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(history_lstm['epoch'], history_lstm['learning_rate'], color='orange', linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('CNN+LSTM: Learning Rate')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/hybrid_DL/CNNLSTM_training_progress.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training plot saved to: analysis/figures/hybrid_DL/CNNLSTM_training_progress.png\n")

# ============================================================================
# TRAINING SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TRAINING SUMMARY - BOTH MODELS")
print("="*70)

# Compare final validation metrics
final_epoch_attn = history_attn['epoch'][-1]
final_epoch_lstm = history_lstm['epoch'][-1]

print(f"\nEpochs Trained:")
print(f"  CNN+Attention: {final_epoch_attn}")
print(f"  CNN+LSTM:      {final_epoch_lstm}")

print(f"\nBest Validation Loss:")
print(f"  CNN+Attention: {min(history_attn['val_loss']):.4f} (epoch {np.argmin(history_attn['val_loss'])+1})")
print(f"  CNN+LSTM:      {min(history_lstm['val_loss']):.4f} (epoch {np.argmin(history_lstm['val_loss'])+1})")

print(f"\nBest Validation R²:")
print(f"  CNN+Attention: {max(history_attn['val_r2']):.4f} (epoch {np.argmax(history_attn['val_r2'])+1})")
print(f"  CNN+LSTM:      {max(history_lstm['val_r2']):.4f} (epoch {np.argmax(history_lstm['val_r2'])+1})")

print(f"\nBest Validation Pearson:")
print(f"  CNN+Attention: {max(history_attn['val_pearson']):.4f} (epoch {np.argmax(history_attn['val_pearson'])+1})")
print(f"  CNN+LSTM:      {max(history_lstm['val_pearson']):.4f} (epoch {np.argmax(history_lstm['val_pearson'])+1})")

print(f"\nModel Checkpoints:")
print(f"  CNN+Attention: {checkpoint_attn}")
print(f"  CNN+LSTM:      {checkpoint_lstm}")

print("\n" + "="*70)
print("✓ BOTH MODELS TRAINED SUCCESSFULLY!")
print("="*70)

# Side-by-side comparison plot
print("\nCreating comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Validation Loss
axes[0, 0].plot(history_attn['epoch'], history_attn['val_loss'], 
                label='CNN+Attention', linewidth=2, color='steelblue')
axes[0, 0].plot(history_lstm['epoch'], history_lstm['val_loss'], 
                label='CNN+LSTM', linewidth=2, color='coral')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Validation Loss')
axes[0, 0].set_title('Validation Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Validation R²
axes[0, 1].plot(history_attn['epoch'], history_attn['val_r2'], 
                label='CNN+Attention', linewidth=2, color='steelblue')
axes[0, 1].plot(history_lstm['epoch'], history_lstm['val_r2'], 
                label='CNN+LSTM', linewidth=2, color='coral')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Validation R²')
axes[0, 1].set_title('Validation R² Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Validation Pearson
axes[1, 0].plot(history_attn['epoch'], history_attn['val_pearson'], 
                label='CNN+Attention', linewidth=2, color='steelblue')
axes[1, 0].plot(history_lstm['epoch'], history_lstm['val_pearson'], 
                label='CNN+LSTM', linewidth=2, color='coral')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Validation Pearson')
axes[1, 0].set_title('Validation Pearson Correlation Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Bar chart of best metrics
metrics_names = ['R²', 'Pearson', 'Spearman']
attn_metrics = [
    max(history_attn['val_r2']),
    max(history_attn['val_pearson']),
    max(history_attn['val_spearman'])
]
lstm_metrics = [
    max(history_lstm['val_r2']),
    max(history_lstm['val_pearson']),
    max(history_lstm['val_spearman'])
]

x = np.arange(len(metrics_names))
width = 0.35

axes[1, 1].bar(x - width/2, attn_metrics, width, label='CNN+Attention', color='steelblue')
axes[1, 1].bar(x + width/2, lstm_metrics, width, label='CNN+LSTM', color='coral')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Best Validation Metrics Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics_names)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('analysis/figures/hybrid_DL/model_comparison_training.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Comparison plot saved to: analysis/figures/hybrid_DL/model_comparison_training.png\n")

print("\n" + "="*70)
print("NEXT STEP: EVALUATION ON TEST SET")
print("="*70)
print("\nNow that both models are trained, we can:")
print("  1. Load the best models")
print("  2. Evaluate on the test set")
print("  3. Create comprehensive visualizations")
print("  4. Compare performance across cell lines")
print("  5. Analyze attention weights (for CNN+Attention)")
print("\nReady to proceed to Step 7 (Evaluation)?\n")

# ============================================================================
# COMPLETE EVALUATION CODE (All functions included)
# ============================================================================

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, loader, dataset, device, set_name='Test'):
    """Comprehensive model evaluation."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {set_name} Set")
    print(f"{'='*70}\n")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_genes = []
    all_cell_lines = []
    
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(loader):
            batch_X = batch_X.to(device)
            predictions = model(batch_X).cpu().numpy()
            
            all_preds.append(predictions)
            all_targets.append(batch_y.numpy())
            
            # Get metadata
            batch_size = len(batch_y)
            start_idx = i * loader.batch_size
            end_idx = start_idx + batch_size
            
            if dataset.genes is not None:
                all_genes.extend(dataset.genes[start_idx:end_idx])
            if dataset.cell_lines is not None:
                all_cell_lines.extend(dataset.cell_lines[start_idx:end_idx])
    
    # Concatenate all batches
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Calculate overall metrics
    metrics = calculate_metrics(targets, predictions)
    
    print(f"Overall Performance:")
    print(f"  RMSE:     {metrics['rmse']:.4f}")
    print(f"  MAE:      {metrics['mae']:.4f}")
    print(f"  R²:       {metrics['r2']:.4f}")
    print(f"  Pearson:  {metrics['pearson']:.4f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    
    # Per-cell-line metrics
    per_cell_line_metrics = {}
    if all_cell_lines:
        unique_cell_lines = np.unique(all_cell_lines)
        print(f"\nPer-Cell-Line Performance:")
        
        for cell_line in unique_cell_lines:
            mask = np.array(all_cell_lines) == cell_line
            cell_preds = predictions[mask]
            cell_targets = targets[mask]
            cell_metrics = calculate_metrics(cell_targets, cell_preds)
            per_cell_line_metrics[cell_line] = cell_metrics
            
            print(f"\n  {cell_line} (n={np.sum(mask):,}):")
            print(f"    R²:      {cell_metrics['r2']:.4f}")
            print(f"    Pearson: {cell_metrics['pearson']:.4f}")
            print(f"    RMSE:    {cell_metrics['rmse']:.4f}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'genes': all_genes,
        'cell_lines': all_cell_lines,
        'metrics': metrics,
        'per_cell_line_metrics': per_cell_line_metrics
    }


# ============================================================================
# LOAD BEST MODELS (FIXED)
# ============================================================================

def load_best_model(model, checkpoint_path, device):
    """Load the best saved model from checkpoint."""
    print(f"\nLoading best model from: {checkpoint_path}")
    
    # FIXED: Add weights_only=False for PyTorch 2.6+
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  ✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"    Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"    Val R²:   {checkpoint['metrics']['r2']:.4f}")
    print(f"    Val Pearson: {checkpoint['metrics']['pearson']:.4f}")
    
    return model, checkpoint

# Load both best models
print("\n" + "="*70)
print("LOADING BEST MODELS")
print("="*70)

cnn_attn_model, checkpoint_attn_info = load_best_model(
    cnn_attn_model, checkpoint_attn, device
)

cnn_lstm_model, checkpoint_lstm_info = load_best_model(
    cnn_lstm_model, checkpoint_lstm, device
)

print("\n✓ Both models loaded successfully!\n")

# ============================================================================
# EVALUATE BOTH MODELS ON TEST SET
# ============================================================================

print("\n" + "="*70)
print("EVALUATING MODELS ON TEST SET")
print("="*70)

# Evaluate CNN+Attention
results_attn = evaluate_model(
    cnn_attn_model,
    loaders['test_loader'],
    loaders['test_dataset'],
    device,
    set_name='Test - CNN+Attention'
)

# Evaluate CNN+LSTM
results_lstm = evaluate_model(
    cnn_lstm_model,
    loaders['test_loader'],
    loaders['test_dataset'],
    device,
    set_name='Test - CNN+LSTM'
)

# ============================================================================
# CREATE COMPREHENSIVE VISUALIZATION
# ============================================================================

def create_comprehensive_plots(results_attn, results_lstm, history_attn, history_lstm):
    """Create all evaluation plots."""
    
    print("\nCreating comprehensive visualization plots...")
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # ====================================================================
    # ROW 1: TRAINING HISTORY
    # ====================================================================
    
    # Training Loss Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history_attn['epoch'], history_attn['train_loss'], 
             label='CNN+Attn Train', linewidth=2, color='steelblue', alpha=0.7)
    ax1.plot(history_attn['epoch'], history_attn['val_loss'], 
             label='CNN+Attn Val', linewidth=2, color='steelblue')
    ax1.plot(history_lstm['epoch'], history_lstm['train_loss'], 
             label='CNN+LSTM Train', linewidth=2, color='coral', alpha=0.7)
    ax1.plot(history_lstm['epoch'], history_lstm['val_loss'], 
             label='CNN+LSTM Val', linewidth=2, color='coral')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Validation R² Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history_attn['epoch'], history_attn['val_r2'], 
             label='CNN+Attention', linewidth=2, color='steelblue')
    ax2.plot(history_lstm['epoch'], history_lstm['val_r2'], 
             label='CNN+LSTM', linewidth=2, color='coral')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.set_title('Validation R²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Validation Pearson
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history_attn['epoch'], history_attn['val_pearson'], 
             label='CNN+Attention', linewidth=2, color='steelblue')
    ax3.plot(history_lstm['epoch'], history_lstm['val_pearson'], 
             label='CNN+LSTM', linewidth=2, color='coral')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Pearson Correlation')
    ax3.set_title('Validation Pearson')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning Rate
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(history_attn['epoch'], history_attn['learning_rate'], 
             label='CNN+Attention', linewidth=2, color='steelblue')
    ax4.plot(history_lstm['epoch'], history_lstm['learning_rate'], 
             label='CNN+LSTM', linewidth=2, color='coral')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ====================================================================
    # ROW 2: TEST SET PREDICTIONS - CNN+ATTENTION
    # ====================================================================
    
    # Overall predictions - CNN+Attention
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.scatter(results_attn['targets'], results_attn['predictions'], 
                alpha=0.5, s=15, edgecolors='k', linewidth=0.3, color='steelblue')
    min_val = min(results_attn['targets'].min(), results_attn['predictions'].min())
    max_val = max(results_attn['targets'].max(), results_attn['predictions'].max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax5.set_xlabel('Actual Expression')
    ax5.set_ylabel('Predicted Expression')
    ax5.set_title(f'CNN+Attention - Overall\nR²={results_attn["metrics"]["r2"]:.4f}')
    ax5.grid(True, alpha=0.3)
    
    # Per-cell-line - CNN+Attention
    cell_lines = np.unique(results_attn['cell_lines'])
    colors_cl = plt.cm.Set3(np.linspace(0, 1, len(cell_lines)))
    
    for idx, (cell_line, color) in enumerate(zip(cell_lines[:3], colors_cl[:3])):
        ax = fig.add_subplot(gs[1, idx+1])
        mask = np.array(results_attn['cell_lines']) == cell_line
        ax.scatter(results_attn['targets'][mask], results_attn['predictions'][mask],
                  alpha=0.6, s=20, color=color, edgecolors='k', linewidth=0.3)
        min_val = results_attn['targets'][mask].min()
        max_val = results_attn['targets'][mask].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        metrics_cl = results_attn['per_cell_line_metrics'][cell_line]
        ax.set_title(f'CNN+Attn - {cell_line}\nR²={metrics_cl["r2"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    # ====================================================================
    # ROW 3: TEST SET PREDICTIONS - CNN+LSTM
    # ====================================================================
    
    # Overall predictions - CNN+LSTM
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.scatter(results_lstm['targets'], results_lstm['predictions'], 
                alpha=0.5, s=15, edgecolors='k', linewidth=0.3, color='coral')
    min_val = min(results_lstm['targets'].min(), results_lstm['predictions'].min())
    max_val = max(results_lstm['targets'].max(), results_lstm['predictions'].max())
    ax9.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax9.set_xlabel('Actual Expression')
    ax9.set_ylabel('Predicted Expression')
    ax9.set_title(f'CNN+LSTM - Overall\nR²={results_lstm["metrics"]["r2"]:.4f}')
    ax9.grid(True, alpha=0.3)
    
    # Per-cell-line - CNN+LSTM
    for idx, (cell_line, color) in enumerate(zip(cell_lines[:3], colors_cl[:3])):
        ax = fig.add_subplot(gs[2, idx+1])
        mask = np.array(results_lstm['cell_lines']) == cell_line
        ax.scatter(results_lstm['targets'][mask], results_lstm['predictions'][mask],
                  alpha=0.6, s=20, color=color, edgecolors='k', linewidth=0.3)
        min_val = results_lstm['targets'][mask].min()
        max_val = results_lstm['targets'][mask].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        metrics_cl = results_lstm['per_cell_line_metrics'][cell_line]
        ax.set_title(f'CNN+LSTM - {cell_line}\nR²={metrics_cl["r2"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    # ====================================================================
    # ROW 4: RESIDUAL ANALYSIS & COMPARISON
    # ====================================================================
    
    # Residuals - CNN+Attention
    ax13 = fig.add_subplot(gs[3, 0])
    residuals_attn = results_attn['targets'] - results_attn['predictions']
    ax13.scatter(results_attn['predictions'], residuals_attn, 
                alpha=0.5, s=15, edgecolors='k', linewidth=0.3, color='steelblue')
    ax13.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax13.set_xlabel('Predicted Expression')
    ax13.set_ylabel('Residuals')
    ax13.set_title('CNN+Attention - Residuals')
    ax13.grid(True, alpha=0.3)
    
    # Residuals - CNN+LSTM
    ax14 = fig.add_subplot(gs[3, 1])
    residuals_lstm = results_lstm['targets'] - results_lstm['predictions']
    ax14.scatter(results_lstm['predictions'], residuals_lstm, 
                alpha=0.5, s=15, edgecolors='k', linewidth=0.3, color='coral')
    ax14.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax14.set_xlabel('Predicted Expression')
    ax14.set_ylabel('Residuals')
    ax14.set_title('CNN+LSTM - Residuals')
    ax14.grid(True, alpha=0.3)
    
    # Metrics Comparison Bar Chart
    ax15 = fig.add_subplot(gs[3, 2])
    metrics_names = ['R²', 'Pearson', 'Spearman']
    attn_vals = [results_attn['metrics']['r2'], 
                 results_attn['metrics']['pearson'],
                 results_attn['metrics']['spearman']]
    lstm_vals = [results_lstm['metrics']['r2'], 
                 results_lstm['metrics']['pearson'],
                 results_lstm['metrics']['spearman']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax15.bar(x - width/2, attn_vals, width, label='CNN+Attention', color='steelblue')
    ax15.bar(x + width/2, lstm_vals, width, label='CNN+LSTM', color='coral')
    ax15.set_ylabel('Score')
    ax15.set_title('Test Set Metrics Comparison')
    ax15.set_xticks(x)
    ax15.set_xticklabels(metrics_names)
    ax15.legend()
    ax15.grid(True, alpha=0.3, axis='y')
    ax15.set_ylim([0, 1])
    
    # Add values on bars
    for i, (v1, v2) in enumerate(zip(attn_vals, lstm_vals)):
        ax15.text(i - width/2, v1 + 0.02, f'{v1:.3f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax15.text(i + width/2, v2 + 0.02, f'{v2:.3f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Error Comparison
    ax16 = fig.add_subplot(gs[3, 3])
    error_metrics = ['RMSE', 'MAE']
    attn_errors = [results_attn['metrics']['rmse'], results_attn['metrics']['mae']]
    lstm_errors = [results_lstm['metrics']['rmse'], results_lstm['metrics']['mae']]
    
    x_err = np.arange(len(error_metrics))
    ax16.bar(x_err - width/2, attn_errors, width, label='CNN+Attention', color='steelblue')
    ax16.bar(x_err + width/2, lstm_errors, width, label='CNN+LSTM', color='coral')
    ax16.set_ylabel('Error')
    ax16.set_title('Test Set Error Comparison')
    ax16.set_xticks(x_err)
    ax16.set_xticklabels(error_metrics)
    ax16.legend()
    ax16.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Complete Model Evaluation - CNN+Attention vs CNN+LSTM', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    save_path = 'analysis/figures/hybrid_DL/complete_evaluation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Complete evaluation plot saved: {save_path}")
    plt.show()

# Create comprehensive plots
create_comprehensive_plots(results_attn, results_lstm, history_attn, history_lstm)

print("\n" + "="*70)
print("✓ STEP 7 COMPLETED - Evaluation finished!")
print("="*70)
print("\nReady for final step (Save Results)? Type 'yes'\n")
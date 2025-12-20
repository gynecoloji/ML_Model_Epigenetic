import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, GridSearchCV
from scipy.stats import spearmanr,pearsonr
import time
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

merged = pd.read_csv("models/data/combined_data.csv",
                      header = 0, sep = ",")
bin_cols = [c for c in merged.columns if "_bin" in c]
X = merged[bin_cols]
y = merged["expression"]
groups = merged["gene_id"]

# First split: Train vs (Val + Test)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
groups_train = groups.iloc[train_idx]

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]
groups_temp = groups.iloc[temp_idx]

# Second split: Val vs Test (50/50 split of the temp set)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))

X_val = X_temp.iloc[val_idx]
y_val = y_temp.iloc[val_idx]
groups_val = groups_temp.iloc[val_idx]

X_test = X_temp.iloc[test_idx]
y_test = y_temp.iloc[test_idx]
groups_test = groups_temp.iloc[test_idx]

# Verify no gene overlap
print(f"\n Split Results:")
print(f"  Train: {len(X_train)} samples, {groups_train.nunique()} genes")
print(f"  Val:   {len(X_val)} samples, {groups_val.nunique()} genes")
print(f"  Test:  {len(X_test)} samples, {groups_test.nunique()} genes")

# Convert to numpy arrays for sklearn
X_train = X_train.values
X_val = X_val.values
X_test = X_test.values
y_train = y_train.values
y_val = y_val.values
y_test = y_test.values

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Train Simple Linear Model
# ═══════════════════════════════════════════════════════════════════════  
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_train_pred_lr = lr_model.predict(X_train)
y_val_pred_lr = lr_model.predict(X_val)
y_test_pred_lr = lr_model.predict(X_test)

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Assess Simple Linear Model
# ═══════════════════════════════════════════════════════════════════════
def evaluate_model(y_true, y_pred, set_name=""):
    """
    Evaluate model performance
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    print(f"  {set_name:5} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson: {pearson_r:.4f}")
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'pearson_r': pearson_r}

lr_train_metrics = evaluate_model(y_train, y_train_pred_lr, "Train")
lr_val_metrics = evaluate_model(y_val, y_val_pred_lr, "Val")
lr_test_metrics = evaluate_model(y_test, y_test_pred_lr, "Test")
overfitting_gap = lr_train_metrics['r2'] - lr_val_metrics['r2']

def plot_predictions(y_true_train, y_pred_train, 
                    y_true_val, y_pred_val, 
                    y_true_test, y_pred_test,
                    model_name="Linear Regression",
                    save_path="analysis/figures/"):
    """
    Create comprehensive prediction plots
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        (y_true_train, y_pred_train, "Train", axes[0]),
        (y_true_val, y_pred_val, "Validation", axes[1]),
        (y_true_test, y_pred_test, "Test", axes[2])
    ]
    
    for y_true, y_pred, name, ax in datasets:
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        pearson_r, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect prediction')
        
        # Labels and title
        ax.set_xlabel('True Expression', fontsize=12)
        ax.set_ylabel('Predicted Expression', fontsize=12)
        ax.set_title(f'{name} Set\nR²={r2:.3f}, Pearson r={pearson_r:.3f}, RMSE={rmse:.3f}', 
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Predicted vs True Expression', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    filename = f"{save_path}{model_name.lower().replace(' ', '_')}_predictions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    return fig

def plot_residuals(y_true_train, y_pred_train, 
                  y_true_val, y_pred_val, 
                  y_true_test, y_pred_test,
                  model_name="Linear Regression",
                  save_path="analysis/figures/"):
    """
    Plot residual analysis
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    datasets = [
        (y_true_train, y_pred_train, "Train"),
        (y_true_val, y_pred_val, "Validation"),
        (y_true_test, y_pred_test, "Test")
    ]
    
    for idx, (y_true, y_pred, name) in enumerate(datasets):
        residuals = y_true - y_pred
        
        # Top row: Residuals vs Predicted
        ax1 = axes[0, idx]
        ax1.scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Expression', fontsize=11)
        ax1.set_ylabel('Residuals', fontsize=11)
        ax1.set_title(f'{name} - Residuals vs Predicted', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Bottom row: Residual distribution
        ax2 = axes[1, idx]
        ax2.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title(f'{name} - Residual Distribution\nMean={residuals.mean():.3f}, Std={residuals.std():.3f}', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_name} - Residual Analysis', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    filename = f"{save_path}{model_name.lower().replace(' ', '_')}_residuals.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    return fig

def plot_performance_comparison(metrics_dict, save_path="analysis/figures/"):
    """
    Compare performance across train/val/test
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    metrics = ['r2', 'rmse', 'mae', 'pearson_r']
    titles = ['R² Score', 'RMSE', 'MAE', 'Pearson Correlation']
    
    sets = ['Train', 'Val', 'Test']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        values = [metrics_dict[s.lower()][metric] for s in sets]
        
        bars = axes[idx].bar(sets, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[idx].set_ylabel(title, fontsize=12, fontweight='bold')
        axes[idx].set_title(title, fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.3f}',
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"{save_path}linear_regression_metrics_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    return fig
  
# 1. Prediction plots
print("\nPredicted vs True plots...")
fig1 = plot_predictions(
    y_train, y_train_pred_lr,
    y_val, y_val_pred_lr,
    y_test, y_test_pred_lr,
    model_name="Linear Regression",
    save_path="analysis/figures/slr/"
)

# 2. Residual plots
print("\nResidual analysis plots...")
fig2 = plot_residuals(
    y_train, y_train_pred_lr,
    y_val, y_val_pred_lr,
    y_test, y_test_pred_lr,
    model_name="Linear Regression",
    save_path="analysis/figures/slr/"
)

# 3. Performance comparison
print("\nPerformance metrics comparison...")
lr_metrics_all = {
    'train': lr_train_metrics,
    'val': lr_val_metrics,
    'test': lr_test_metrics
}
fig3 = plot_performance_comparison(lr_metrics_all,
                                   save_path="analysis/figures/slr/")

print("\n✅ All visualizations completed!")


def print_detailed_assessment(train_metrics, val_metrics, test_metrics, model_name="Linear Regression"):
    """
    Print comprehensive model assessment
    """
    print("\n" + "="*70)
    print(f"{model_name.upper()} - DETAILED ASSESSMENT")
    print("="*70)
    
    print("\nPERFORMANCE SUMMARY")
    print("-" * 70)
    print(f"{'Metric':<20} {'Train':<15} {'Validation':<15} {'Test':<15}")
    print("-" * 70)
    print(f"{'R² Score':<20} {train_metrics['r2']:<15.4f} {val_metrics['r2']:<15.4f} {test_metrics['r2']:<15.4f}")
    print(f"{'RMSE':<20} {train_metrics['rmse']:<15.4f} {val_metrics['rmse']:<15.4f} {test_metrics['rmse']:<15.4f}")
    print(f"{'MAE':<20} {train_metrics['mae']:<15.4f} {val_metrics['mae']:<15.4f} {test_metrics['mae']:<15.4f}")
    print(f"{'Pearson r':<20} {train_metrics['pearson_r']:<15.4f} {val_metrics['pearson_r']:<15.4f} {test_metrics['pearson_r']:<15.4f}")
    
    print("\nOVERFITTING ANALYSIS")
    print("-" * 70)
    train_val_gap = train_metrics['r2'] - val_metrics['r2']
    train_test_gap = train_metrics['r2'] - test_metrics['r2']
    
    print(f"  Train R² - Val R²:  {train_val_gap:.4f}")
    print(f"  Train R² - Test R²: {train_test_gap:.4f}")
    
    if train_val_gap > 0.15:
        print("  WARNING: Significant overfitting detected!")
        print("     → Consider regularization (Ridge/Lasso)")
        print("     → Or use ensemble methods (Random Forest)")
    elif train_val_gap > 0.05:
        print("  Moderate overfitting - regularization recommended")
    else:
        print("  Good generalization!")
    
    print("\nPERFORMANCE INTERPRETATION")
    print("-" * 70)
    
    val_r2 = val_metrics['r2']
    if val_r2 > 0.7:
        print("  EXCELLENT: Strong predictive power (R² > 0.7)")
    elif val_r2 > 0.5:
        print("  GOOD: Moderate predictive power (R² > 0.5)")
    elif val_r2 > 0.3:
        print("  FAIR: Modest predictive power (R² > 0.3)")
    else:
        print("  WEAK: Limited predictive power (R² < 0.3)")
        print("     → Try more complex models")
        print("     → Consider feature engineering")
    
    print("\nNEXT STEPS")
    print("-" * 70)
    if train_val_gap > 0.1:
        print("  → Try Ridge Regression (L2 regularization)")
        print("  → Try Lasso Regression (L1 regularization for feature selection)")
    print("  → Try Random Forest (captures non-linear patterns)")
    print("  → Consider dimensionality reduction (PCA)")
    
    print("\n" + "="*70)

print_detailed_assessment(lr_train_metrics, lr_val_metrics, lr_test_metrics)
with open('models/trained/linear_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
    
# Train R²: 0.6-0.8 (likely high due to many features)
# Val R²: 0.3-0.5 (realistic performance on unseen genes)
# Overfitting gap: Large (this is expected with 10,000 features)

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Interpret Simple Linear Model
# ═══════════════════════════════════════════════════════════════════════
def visualize_coefficient_patterns(model, feature_names, save_path="analysis/figures/"):
    """
    Visualize coefficient patterns to interpret what the model learned
    Updated for: 5% gene body upstream + gene body + 5% gene body downstream
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*70)
    print("COEFFICIENT PATTERN ANALYSIS")
    print("="*70)
    
    # Extract coefficients
    coefficients = model.coef_
    
    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Extract histone mark (everything before "_bin")
    coef_df['Histone_Mark'] = coef_df['Feature'].str.split('_bin').str[0]
    
    # Extract bin number
    coef_df['Bin'] = coef_df['Feature'].str.extract(r'_bin(\d+)')[0].astype(int)
    
    print(f"\n  Extracted histone marks: {coef_df['Histone_Mark'].unique().tolist()}")
    print(f"  Bin range: {coef_df['Bin'].min()} to {coef_df['Bin'].max()}")
    
    # Pivot to create matrix: Histone_Mark × Bin
    coef_matrix = coef_df.pivot(index='Histone_Mark', columns='Bin', values='Coefficient')
    
    print(f"  ✓ Coefficient matrix shape: {coef_matrix.shape}")
    print(f"  ✓ Histone marks in analysis: {list(coef_matrix.index)}")
    
    # Calculate region boundaries (5% upstream, 100% gene body, 5% downstream)
    total_bins = coef_matrix.shape[1]
    upstream_end = int(total_bins * 5/110)  # First 5/110 of bins
    downstream_start = int(total_bins * 105/110)  # Last 5/110 of bins
    
    print(f"\n  Region boundaries:")
    print(f"    Upstream (5% gene length): bins 1-{upstream_end}")
    print(f"    Gene Body (100% length): bins {upstream_end+1}-{downstream_start}")
    print(f"    Downstream (5% gene length): bins {downstream_start+1}-{total_bins}")
    
    return coef_df, coef_matrix, upstream_end, downstream_start
  
def plot_coefficient_heatmap(coef_matrix, upstream_end, downstream_start, 
                            save_path="analysis/figures/"):
    """
    Create heatmap of coefficients across bins and histone marks
    Shows SIGNED coefficients (positive and negative values)
    Updated for: 5% gene body upstream + gene body + 5% gene body downstream
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nCreating coefficient heatmap (signed values)...")
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Create heatmap with signed values
    sns.heatmap(coef_matrix, 
                cmap='RdBu_r',  # Red = negative, Blue = positive
                center=0,
                cbar_kws={'label': 'Coefficient Value (Signed)'},
                xticklabels=50,  # Show every 50th bin
                yticklabels=True,
                ax=ax,
                vmin=coef_matrix.min().min(),  # Use actual min/max
                vmax=coef_matrix.max().max())
    
    # Add region boundary markers
    ax.axvline(x=upstream_end, color='yellow', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(x=downstream_start, color='yellow', linestyle='--', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Bin Position (5% Upstream ← Gene Body (100%) → 5% Downstream)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Histone Mark', fontsize=14, fontweight='bold')
    
    # Add text annotations for regions
    ax.text(upstream_end/2, -0.5, 'Upstream\n(5% gene length)', 
           ha='center', fontsize=10, fontweight='bold', color='darkblue')
    ax.text((upstream_end + downstream_start)/2, -0.5, 'Gene Body\n(100% length)', 
           ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    ax.text((downstream_start + coef_matrix.shape[1])/2, -0.5, 'Downstream\n(5% gene length)', 
           ha='center', fontsize=10, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    filename = f"{save_path}lr_coefficient_heatmap_signed.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
def plot_coefficient_profiles(coef_matrix, upstream_end, downstream_start, 
                              save_path="analysis/figures/"):
    """
    Plot coefficient profiles across gene region for each histone mark
    Shows SIGNED coefficients (positive and negative values)
    Updated for: 5% gene body upstream + gene body + 5% gene body downstream
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nCreating coefficient profiles (signed values)...")
    
    fig, axes = plt.subplots(len(coef_matrix), 1, 
                            figsize=(16, 3*len(coef_matrix)), sharex=True)
    
    # If only one histone mark, make axes a list
    if len(coef_matrix) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(range(len(coef_matrix)))
    
    total_bins = coef_matrix.shape[1]
    legend_handles = []
    legend_labels = []
    for idx, (mark, color) in enumerate(zip(coef_matrix.index, colors)):
        ax = axes[idx]
        
        # Plot coefficient profile (SIGNED values)
        bins = coef_matrix.columns
        values = coef_matrix.loc[mark].values
        
        ax.plot(bins, values, color=color, linewidth=2.5, alpha=0.9)
        ax.fill_between(bins, 0, values, where=(values > 0), 
                        color=color, alpha=0.3, label='Positive effect (↑ expression)')
        ax.fill_between(bins, 0, values, where=(values < 0), 
                        color='red', alpha=0.3, label='Negative effect (↓ expression)')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Mark regions with new boundaries
        ax.axvspan(0, upstream_end, alpha=0.1, color='blue', label='Upstream (5%)')
        ax.axvspan(upstream_end, downstream_start, alpha=0.1, color='green')
        ax.axvspan(downstream_start, total_bins, alpha=0.1, color='orange', 
                  label='Downstream (5%)')
        
        # Add region boundaries
        ax.axvline(x=upstream_end, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(x=downstream_start, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        
        # Labels
        ax.set_ylabel('Coefficient', fontsize=11, fontweight='bold')
        ax.set_title(f'{mark}', fontsize=13, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        mean_coef = values.mean()
        max_coef = values.max()
        min_coef = values.min()
        textstr = f'Mean: {mean_coef:.5f}\nMax: {max_coef:.5f}\nMin: {min_coef:.5f}'
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles = handles
            legend_labels = labels
    
    axes[-1].set_xlabel('Bin Position (5% Upstream ← Gene Body (100%) → 5% Downstream)', 
                       fontsize=12, fontweight='bold')
    fig.legend(
      legend_handles,
      legend_labels,
      loc="upper center",
      ncol=4,
      frameon=False,
      fontsize=10,
      bbox_to_anchor=(0.5, 1.02)
    )
    plt.tight_layout()
    
    filename = f"{save_path}lr_coefficient_profiles_signed.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
def plot_regional_importance(coef_matrix, upstream_end, downstream_start, 
                            save_path="analysis/figures/"):
    """
    Analyze importance of different gene regions
    Shows BOTH signed coefficients and absolute values
    Updated for: 5% gene body upstream + gene body + 5% gene body downstream
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nAnalyzing regional importance (signed & absolute)...")
    
    total_bins = coef_matrix.shape[1]
    
    # Define regions based on new setup
    # 5% upstream + 100% gene body + 5% downstream
    regions = {
        'Upstream': (1, upstream_end),
        'TSS Region (0%-10%)': (upstream_end, int(upstream_end + (downstream_start - upstream_end) * 0.1)),
        'Gene Body (10%-33%)': (int(upstream_end + (downstream_start - upstream_end) * 0.1), 
                               int(upstream_end + (downstream_start - upstream_end) * 0.33)),
        'Gene Body (33%-66%)': (int(upstream_end + (downstream_start - upstream_end) * 0.33), 
                                int(upstream_end + (downstream_start - upstream_end) * 0.66)),
        'Gene Body (66%-100%)': (int(upstream_end + (downstream_start - upstream_end) * 0.66), 
                                downstream_start),
        'Downstream': (downstream_start, total_bins)
    }
    
    # Calculate SIGNED mean coefficient for each region
    regional_signed = {}
    regional_absolute = {}
    
    for mark in coef_matrix.index:
        regional_signed[mark] = {}
        regional_absolute[mark] = {}
        for region_name, (start, end) in regions.items():
            region_coefs = coef_matrix.loc[mark, start:end]
            mean_signed_coef = region_coefs.mean()  # SIGNED
            mean_abs_coef = region_coefs.abs().mean()  # ABSOLUTE
            regional_signed[mark][region_name] = mean_signed_coef
            regional_absolute[mark][region_name] = mean_abs_coef
    
    # Convert to dataframes
    signed_df = pd.DataFrame(regional_signed).T
    absolute_df = pd.DataFrame(regional_absolute).T
    
    print("\n  Regional SIGNED coefficients (mean):")
    print(signed_df.round(6))
    print("\n  Regional ABSOLUTE coefficients (mean):")
    print(absolute_df.round(6))
    
    # Plot both signed and absolute
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: SIGNED coefficients (shows direction)
    signed_df.plot(kind='bar', ax=axes[0], width=0.8, 
                   color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#ee5a6f', '#c8d6e5'])
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean Signed Coefficient', fontsize=12, fontweight='bold')
    axes[0].set_title('A. Regional Analysis - SIGNED Coefficients\n(Positive = increases expression, Negative = decreases expression)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(title='Gene Region', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: ABSOLUTE coefficients (shows magnitude)
    absolute_df.plot(kind='bar', ax=axes[1], width=0.8,
                    color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#ee5a6f', '#c8d6e5'])
    axes[1].set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Coefficient', fontsize=12, fontweight='bold')
    axes[1].set_title('B. Regional Analysis - ABSOLUTE Coefficients\n(Shows magnitude of effect regardless of direction)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(title='Gene Region', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    filename = f"{save_path}lr_regional_importance_signed_and_absolute.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    return signed_df, absolute_df
  
def plot_histone_mark_importance(coef_df, save_path="analysis/figures/"):
    """
    Compare overall importance of different histone marks
    Shows BOTH signed mean and absolute mean coefficients
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nComparing histone mark importance (signed & absolute)...")
    
    # Calculate SIGNED mean coefficient by histone mark
    mark_signed_importance = coef_df.groupby('Histone_Mark')['Coefficient'].mean().reset_index()
    mark_signed_importance.columns = ['Histone_Mark', 'Mean_Signed_Coef']
    
    # Calculate ABSOLUTE mean coefficient by histone mark
    mark_abs_importance = coef_df.groupby('Histone_Mark')['Coefficient'].apply(
        lambda x: x.abs().mean()
    ).reset_index()
    mark_abs_importance.columns = ['Histone_Mark', 'Mean_Abs_Coef']
    
    # Merge both
    mark_importance = pd.merge(mark_signed_importance, mark_abs_importance, on='Histone_Mark')
    
    # Sort by absolute value for ranking
    mark_importance = mark_importance.sort_values('Mean_Abs_Coef', ascending=False)
    
    print("\n  Histone mark importance (both signed and absolute):")
    print(mark_importance)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    colors = plt.cm.Set3(range(len(mark_importance)))
    
    # Plot 1: SIGNED mean coefficient (shows direction)
    bars1 = axes[0, 0].bar(range(len(mark_importance)), 
                           mark_importance['Mean_Signed_Coef'],
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    axes[0, 0].set_xticks(range(len(mark_importance)))
    axes[0, 0].set_xticklabels(mark_importance['Histone_Mark'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mean Signed Coefficient', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('A. Overall Effect (SIGNED)\n(Positive = activating, Negative = repressive)', 
                         fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars1, mark_importance['Mean_Signed_Coef'])):
        height = bar.get_height()
        y_pos = height if height > 0 else height
        va = 'bottom' if height > 0 else 'top'
        axes[0, 0].text(i, y_pos, f'{val:.6f}',
                       ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Plot 2: ABSOLUTE mean coefficient (shows magnitude)
    bars2 = axes[0, 1].bar(range(len(mark_importance)), 
                           mark_importance['Mean_Abs_Coef'],
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[0, 1].set_xticks(range(len(mark_importance)))
    axes[0, 1].set_xticklabels(mark_importance['Histone_Mark'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Mean Absolute Coefficient', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('B. Overall Importance (ABSOLUTE)\n(Ranked by predictive contribution)', 
                         fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars2, mark_importance['Mean_Abs_Coef'])):
        height = bar.get_height()
        axes[0, 1].text(i, height, f'{val:.6f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Distribution of SIGNED coefficients by histone mark
    unique_marks = mark_importance['Histone_Mark'].tolist()
    for i, mark in enumerate(unique_marks):
        mark_coefs = coef_df[coef_df['Histone_Mark'] == mark]['Coefficient']
        axes[1, 0].hist(mark_coefs, bins=50, alpha=0.6, label=mark, color=colors[i])
    
    axes[1, 0].set_xlabel('Coefficient Value (Signed)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('C. Coefficient Distribution (SIGNED)', fontsize=13, fontweight='bold')
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero')
    axes[1, 0].legend(fontsize=9, loc='best')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Comparison of signed vs absolute
    x = np.arange(len(mark_importance))
    width = 0.35
    
    bars_signed = axes[1, 1].bar(x - width/2, mark_importance['Mean_Signed_Coef'], 
                                width, label='Signed Mean', color='steelblue', 
                                alpha=0.8, edgecolor='black')
    bars_abs = axes[1, 1].bar(x + width/2, mark_importance['Mean_Abs_Coef'], 
                             width, label='Absolute Mean', color='coral', 
                             alpha=0.8, edgecolor='black')
    
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(mark_importance['Histone_Mark'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('D. Signed vs Absolute Comparison', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Histone Mark Importance Analysis: Signed and Absolute Values', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_path}lr_histone_importance_signed_and_absolute.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    return mark_importance

def create_interpretation_summary(coef_matrix, signed_df, absolute_df, mark_importance, 
                                 upstream_end, downstream_start,
                                 save_path="analysis/figures/"):
    """
    Create a comprehensive interpretation summary figure
    Shows SIGNED coefficients throughout
    Updated for: 5% gene body upstream + gene body + 5% gene body downstream
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nCreating comprehensive interpretation summary (signed values)...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    total_bins = coef_matrix.shape[1]
    
    # ═══════════════════════════════════════════════════════════
    # Top: Heatmap (spans both columns) - SIGNED
    # ═══════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(coef_matrix, cmap='RdBu_r', center=0,
                xticklabels=50, yticklabels=True, ax=ax1,
                cbar_kws={'label': 'Coefficient (Signed)'})
    ax1.axvline(x=upstream_end, color='yellow', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.axvline(x=downstream_start, color='yellow', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.set_title('A. Coefficient Patterns (SIGNED) Across Gene Region\n5% Upstream + Gene Body + 5% Downstream', 
                 fontsize=14, fontweight='bold', loc='left')
    ax1.set_xlabel('Bin Position (5% Upstream ← Gene Body → 5% Downstream)')
    ax1.set_ylabel('Histone Mark')
    
    # ═══════════════════════════════════════════════════════════
    # Second row left: Regional importance - SIGNED
    # ═══════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    signed_df.plot(kind='bar', ax=ax2, width=0.7, legend=False)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.set_title('B. Regional Analysis (SIGNED)\nPositive = activating, Negative = repressive', 
                 fontsize=13, fontweight='bold', loc='left')
    ax2.set_xlabel('Histone Mark', fontsize=11)
    ax2.set_ylabel('Mean Signed Coefficient', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='right', fontsize=9)
    
    # ═══════════════════════════════════════════════════════════
    # Second row right: Regional importance - ABSOLUTE
    # ═══════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    absolute_df.plot(kind='bar', ax=ax3, width=0.7, legend=False)
    ax3.set_title('C. Regional Analysis (ABSOLUTE)\nMagnitude of effect', 
                 fontsize=13, fontweight='bold', loc='left')
    ax3.set_xlabel('Histone Mark', fontsize=11)
    ax3.set_ylabel('Mean Absolute Coefficient', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='right', fontsize=9)
    
    # ═══════════════════════════════════════════════════════════
    # Third row left: Overall histone mark importance - SIGNED
    # ═══════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, 0])
    colors = plt.cm.Set3(range(len(mark_importance)))
    bars_signed = ax4.bar(range(len(mark_importance)), mark_importance['Mean_Signed_Coef'],
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax4.set_xticks(range(len(mark_importance)))
    ax4.set_xticklabels(mark_importance['Histone_Mark'], rotation=0, ha='right', fontsize=9)
    ax4.set_title('D. Overall Effect by Mark (SIGNED)', 
                 fontsize=13, fontweight='bold', loc='left')
    ax4.set_xlabel('Histone Mark', fontsize=11)
    ax4.set_ylabel('Mean Signed Coefficient', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, val in enumerate(mark_importance['Mean_Signed_Coef']):
        height = val
        y_pos = height if height > 0 else height
        va = 'bottom' if height > 0 else 'top'
        ax4.text(i, y_pos, f'{val:.5f}', ha='center', va=va, 
                fontsize=8, fontweight='bold')
    
    # ═══════════════════════════════════════════════════════════
    # Third row right: Overall histone mark importance - ABSOLUTE
    # ═══════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 1])
    bars_abs = ax5.bar(range(len(mark_importance)), mark_importance['Mean_Abs_Coef'],
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_xticks(range(len(mark_importance)))
    ax5.set_xticklabels(mark_importance['Histone_Mark'], rotation=0, ha='right', fontsize=9)
    ax5.set_title('E. Overall Importance by Mark (ABSOLUTE)\nRanked by magnitude', 
                 fontsize=13, fontweight='bold', loc='left')
    ax5.set_xlabel('Histone Mark', fontsize=11)
    ax5.set_ylabel('Mean Absolute Coefficient', fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, val in enumerate(mark_importance['Mean_Abs_Coef']):
        ax5.text(i, val, f'{val:.5f}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
    
    # ═══════════════════════════════════════════════════════════
    # Bottom: Coefficient profiles (spans both columns) - SIGNED
    # ═══════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[3, :])
    plot_colors = plt.cm.Set2(range(len(coef_matrix)))
    for mark, color in zip(coef_matrix.index, plot_colors):
        ax6.plot(coef_matrix.columns, coef_matrix.loc[mark], 
                label=mark, color=color, linewidth=2.5, alpha=0.9)
    
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax6.axvspan(0, upstream_end, alpha=0.1, color='blue')
    ax6.axvspan(upstream_end, downstream_start, alpha=0.1, color='green')
    ax6.axvspan(downstream_start, total_bins, alpha=0.1, color='orange')
    ax6.axvline(x=upstream_end, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax6.axvline(x=downstream_start, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    
    # Add region labels
    ax6.text(upstream_end/2, ax6.get_ylim()[0]*0.9, 'Upstream\n(5%)', 
            ha='center', fontsize=10, fontweight='bold', color='darkblue')
    ax6.text((upstream_end + downstream_start)/2, ax6.get_ylim()[0]*0.9, 'Gene Body\n(100%)', 
            ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    ax6.text((downstream_start + total_bins)/2, ax6.get_ylim()[0]*0.9, 'Downstream\n(5%)', 
            ha='center', fontsize=10, fontweight='bold', color='darkred')
    
    ax6.set_title('F. Coefficient Profiles (SIGNED) Across Gene Region', 
                 fontsize=14, fontweight='bold', loc='left')
    ax6.set_xlabel('Bin Position (5% Upstream ← Gene Body (100%) → 5% Downstream)', 
                  fontsize=12)
    ax6.set_ylabel('Coefficient', fontsize=12)
    ax6.legend(loc='upper right', ncol=3, fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Linear Regression: Model Interpretation Summary (Signed Coefficients)\n5% Gene Length Upstream + Gene Body + 5% Gene Length Downstream', 
                fontsize=18, fontweight='bold')
    
    filename = f"{save_path}lr_interpretation_summary_signed.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()

coef_df, coef_matrix, upstream_end, downstream_start = visualize_coefficient_patterns(
    model=lr_model,
    feature_names=bin_cols,
    save_path="analysis/figures/slr"
)

# This returns:
# - coef_df: DataFrame with Feature, Coefficient, Histone_Mark, Bin columns
# - coef_matrix: Pivot table (Histone_Mark × Bin)
# - upstream_end: Bin number where upstream region ends (5% boundary)
# - downstream_start: Bin number where downstream region starts (105% boundary)

plot_coefficient_heatmap(
    coef_matrix=coef_matrix,
    upstream_end=upstream_end,
    downstream_start=downstream_start,
    save_path="analysis/figures/slr/"
)

plot_coefficient_profiles(
    coef_matrix=coef_matrix,
    upstream_end=upstream_end,
    downstream_start=downstream_start,
    save_path="analysis/figures/slr/"
)

signed_df, absolute_df = plot_regional_importance(
    coef_matrix=coef_matrix,
    upstream_end=upstream_end,
    downstream_start=downstream_start,
    save_path="analysis/figures/slr/"
)

# This returns:
# - signed_df: Mean signed coefficient by region and histone mark
# - absolute_df: Mean absolute coefficient by region and histone mark

mark_importance = plot_histone_mark_importance(
    coef_df=coef_df,
    save_path="analysis/figures/slr/"
)

# This returns:
# - mark_importance: DataFrame with Histone_Mark, Mean_Signed_Coef, Mean_Abs_Coef

create_interpretation_summary(
    coef_matrix=coef_matrix,
    signed_df=signed_df,
    absolute_df=absolute_df,
    mark_importance=mark_importance,
    upstream_end=upstream_end,
    downstream_start=downstream_start,
    save_path="analysis/figures/slr/"
)

# Save coefficient dataframe
os.makedirs("models/results/slr/")
coef_df.to_csv("models/results/slr/coefficient_values.csv", index=False)

# Save coefficient matrix
coef_matrix.to_csv("models/results/slr/coefficient_matrix.csv")

# Save regional importance (signed)
signed_df.to_csv("models/results/slr/regional_importance_signed.csv")

# Save regional importance (absolute)
absolute_df.to_csv("models/results/slr/regional_importance_absolute.csv")

# Save histone mark importance
mark_importance.to_csv("models/results/slr/histone_mark_importance.csv", index=False)

print("\nAll visualizations and data exports complete!")
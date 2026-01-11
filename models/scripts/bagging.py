import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr, spearmanr
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import time

os.chdir("ML")
df = pd.read_csv("models/data/combined_data.csv")

# Assuming your dataframe is called 'df'
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist()[:10], "...")  # Show first 10 columns

# Check data structure
print("\n=== Data Structure ===")
print(f"Unique genes: {df['gene_id'].nunique()}")
print(f"Unique cell lines: {df['cell_line'].nunique()}")
print(f"Cell lines: {df['cell_line'].unique()}")

# Check observations per gene
print("\n=== Observations per gene ===")
obs_per_gene = df.groupby('gene_id').size()
print(obs_per_gene.value_counts().sort_index())

# Identify feature columns (ChIP-seq bins)
feature_cols = [col for col in df.columns 
                if col not in ['gene_id', 'cell_line', 'expression']]
print(f"\nTotal features: {len(feature_cols)}")

# Check for missing values
print(f"\nMissing values in expression: {df['expression'].isna().sum()}")
print(f"Missing values in features: {df[feature_cols].isna().sum().sum()}")

# ===== Section 2: Gene-based Train-Test Split =====

# Prepare features (X) and target (y)
X = df[feature_cols].values
y = df['expression'].values
groups = df['gene_id'].values  # Gene IDs for grouping

print("=== Gene-based Data Splitting ===")

# Create gene-based train-test split (80/20)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

# Split the data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

# Verify no gene leakage
train_genes = set(groups_train)
test_genes = set(groups_test)
overlap = train_genes & test_genes

print(f"\nTrain set: {len(X_train)} observations, {len(train_genes)} genes")
print(f"Test set:  {len(X_test)} observations, {len(test_genes)} genes")
print(f"Gene overlap: {len(overlap)} (should be 0!)")

# Verify each gene's observations stay together
print("\n=== Verification: Gene distribution ===")
print("Sample of train genes and their counts:")
train_df = df.iloc[train_idx]
print(train_df.groupby('gene_id').size().head())

assert len(overlap) == 0, "ERROR: Gene leakage detected!"
print("\n✓ Success: No gene leakage - split is valid!")

# ===== Section 3: ENHANCED Comprehensive Parameter Assessment =====

print("="*80)
print("ENHANCED COMPREHENSIVE BAGGING CONFIGURATION ASSESSMENT")
print("="*80)

# Define parameter combinations to test
param_configs = {
    'Config_1_VeryConservative': {
        'base': {'max_depth': 6, 'min_samples_split': 150, 'min_samples_leaf': 75, 'max_features': 0.5},
        'bagging': {'n_estimators': 100, 'max_samples': 0.7, 'max_features': 0.7}
    },
    'Config_2_Conservative': {
        'base': {'max_depth': 8, 'min_samples_split': 100, 'min_samples_leaf': 50, 'max_features': 0.6},
        'bagging': {'n_estimators': 100, 'max_samples': 0.8, 'max_features': 0.8}
    },
    'Config_3_Moderate': {
        'base': {'max_depth': 10, 'min_samples_split': 50, 'min_samples_leaf': 25, 'max_features': 0.7},
        'bagging': {'n_estimators': 100, 'max_samples': 0.8, 'max_features': 0.8}
    },
    'Config_4_Balanced': {
        'base': {'max_depth': 12, 'min_samples_split': 30, 'min_samples_leaf': 15, 'max_features': 0.8},
        'bagging': {'n_estimators': 100, 'max_samples': 0.7, 'max_features': 0.9}
    },
    'Config_5_LessRegularized': {
        'base': {'max_depth': 15, 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 0.9},
        'bagging': {'n_estimators': 100, 'max_samples': 0.8, 'max_features': 1.0}
    },
    'Config_6_MoreTrees': {
        'base': {'max_depth': 10, 'min_samples_split': 50, 'min_samples_leaf': 25, 'max_features': 0.7},
        'bagging': {'n_estimators': 150, 'max_samples': 0.7, 'max_features': 0.8}
    },
    'Config_7_DeepWithConstraints': {
        'base': {'max_depth': 20, 'min_samples_split': 40, 'min_samples_leaf': 20, 'max_features': 0.8},
        'bagging': {'n_estimators': 80, 'max_samples': 0.75, 'max_features': 0.85}
    },
}

# Store comprehensive results
results = []
per_cell_line_results = []

# Get cell line info for test set
test_cell_lines = df.iloc[test_idx]['cell_line'].values
unique_cell_lines = sorted(df['cell_line'].unique())

# Test each configuration
for config_name, params in param_configs.items():
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    
    # Display parameters compactly
    print(f"Base Tree: depth={params['base']['max_depth']}, "
          f"split={params['base']['min_samples_split']}, "
          f"leaf={params['base']['min_samples_leaf']}, "
          f"feats={params['base']['max_features']}")
    print(f"Bagging:   n_est={params['bagging']['n_estimators']}, "
          f"samples={params['bagging']['max_samples']}, "
          f"feats={params['bagging']['max_features']}")
    
    # Create and train model
    base_est = DecisionTreeRegressor(**params['base'], random_state=42)
    model = BaggingRegressor(
        estimator=base_est,
        **params['bagging'],
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # ===== CRITERION 1: Overall Performance =====
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # ===== CRITERION 2: Correlation Coefficients =====
    # Pearson correlation (linear relationship)
    train_pearson_r, train_pearson_p = pearsonr(y_train, y_train_pred)
    test_pearson_r, test_pearson_p = pearsonr(y_test, y_test_pred)
    
    # Spearman correlation (monotonic relationship, rank-based)
    train_spearman_r, train_spearman_p = spearmanr(y_train, y_train_pred)
    test_spearman_r, test_spearman_p = spearmanr(y_test, y_test_pred)
    
    # ===== CRITERION 3: Generalization Quality =====
    r2_gap = train_r2 - test_r2
    r2_ratio = test_r2 / train_r2 if train_r2 > 0 else 0
    rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')
    pearson_gap = train_pearson_r - test_pearson_r
    
    # Generalization score (closer to 1 is better)
    generalization_score = 1 - abs(r2_gap) if abs(r2_gap) < 1 else 0
    
    # ===== CRITERION 4: Prediction Stability =====
    residuals_test = y_test - y_test_pred
    residuals_std = np.std(residuals_test)
    residuals_mean_abs = np.mean(np.abs(residuals_test))
    
    # ===== CRITERION 5: Per-Cell Line Assessment =====
    cell_line_metrics = {}
    
    print(f"\n📊 Per-Cell Line Performance:")
    print(f"{'Cell Line':<12} {'N':<6} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'Pearson':<10} {'Spearman':<10}")
    print("-" * 70)
    
    for cell_line in unique_cell_lines:
        mask = test_cell_lines == cell_line
        n_samples = mask.sum()
        
        if n_samples > 0:
            # Calculate metrics
            cl_y_true = y_test[mask]
            cl_y_pred = y_test_pred[mask]
            
            cl_r2 = r2_score(cl_y_true, cl_y_pred)
            cl_rmse = np.sqrt(mean_squared_error(cl_y_true, cl_y_pred))
            cl_mae = mean_absolute_error(cl_y_true, cl_y_pred)
            cl_pearson_r, cl_pearson_p = pearsonr(cl_y_true, cl_y_pred)
            cl_spearman_r, cl_spearman_p = spearmanr(cl_y_true, cl_y_pred)
            
            # Store metrics
            cell_line_metrics[cell_line] = {
                'n_samples': n_samples,
                'r2': cl_r2,
                'rmse': cl_rmse,
                'mae': cl_mae,
                'pearson_r': cl_pearson_r,
                'pearson_p': cl_pearson_p,
                'spearman_r': cl_spearman_r,
                'spearman_p': cl_spearman_p
            }
            
            # Print row
            print(f"{cell_line:<12} {n_samples:<6} {cl_r2:<8.4f} {cl_rmse:<8.4f} "
                  f"{cl_mae:<8.4f} {cl_pearson_r:<10.4f} {cl_spearman_r:<10.4f}")
            
            # Store per-cell line results for later analysis
            per_cell_line_results.append({
                'Configuration': config_name,
                'Cell_Line': cell_line,
                'N_Samples': n_samples,
                'R2': cl_r2,
                'RMSE': cl_rmse,
                'MAE': cl_mae,
                'Pearson_R': cl_pearson_r,
                'Spearman_R': cl_spearman_r
            })
    
    # Calculate consistency metrics across cell lines
    r2_values = [m['r2'] for m in cell_line_metrics.values()]
    pearson_values = [m['pearson_r'] for m in cell_line_metrics.values()]
    
    r2_mean_across_lines = np.mean(r2_values)
    r2_std_across_lines = np.std(r2_values)
    r2_min_across_lines = np.min(r2_values)
    r2_max_across_lines = np.max(r2_values)
    
    pearson_mean_across_lines = np.mean(pearson_values)
    pearson_std_across_lines = np.std(pearson_values)
    pearson_min_across_lines = np.min(pearson_values)
    
    # ===== CRITERION 6: Efficiency =====
    time_per_tree = train_time / params['bagging']['n_estimators']
    
    # ===== CRITERION 7: Composite Scores =====
    # Score 1: Balance performance and generalization
    composite_score_1 = test_r2 - (r2_gap * 0.5)
    
    # Score 2: Weighted multi-objective score
    # Weights: 35% test R², 25% test Pearson, 20% low gap, 15% consistency, 5% efficiency
    norm_test_r2 = test_r2
    norm_test_pearson = (test_pearson_r + 1) / 2  # Convert -1 to 1 → 0 to 1
    norm_gap = max(0, 1 - (r2_gap / 0.5))
    norm_consistency = max(0, 1 - (r2_std_across_lines / 0.3))
    norm_efficiency = max(0, 1 - (train_time / 300))
    
    composite_score_2 = (0.35 * norm_test_r2 + 
                         0.25 * norm_test_pearson +
                         0.20 * norm_gap + 
                         0.15 * norm_consistency + 
                         0.05 * norm_efficiency)
    
    # ===== Store All Results =====
    result_dict = {
        'Configuration': config_name,
        # Overall Performance
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
        # Correlation
        'Train_Pearson_R': train_pearson_r,
        'Test_Pearson_R': test_pearson_r,
        'Train_Spearman_R': train_spearman_r,
        'Test_Spearman_R': test_spearman_r,
        # Generalization
        'R2_Gap': r2_gap,
        'Pearson_Gap': pearson_gap,
        'R2_Ratio': r2_ratio,
        'RMSE_Ratio': rmse_ratio,
        'Generalization_Score': generalization_score,
        # Stability
        'Residual_Std': residuals_std,
        'Residual_Mean_Abs': residuals_mean_abs,
        # Cell Line Consistency
        'R2_Mean_CellLines': r2_mean_across_lines,
        'R2_Std_CellLines': r2_std_across_lines,
        'R2_Min_CellLine': r2_min_across_lines,
        'R2_Max_CellLine': r2_max_across_lines,
        'Pearson_Mean_CellLines': pearson_mean_across_lines,
        'Pearson_Std_CellLines': pearson_std_across_lines,
        'Pearson_Min_CellLine': pearson_min_across_lines,
        # Efficiency
        'Train_Time_sec': train_time,
        'Time_Per_Tree': time_per_tree,
        # Composite
        'Composite_Score_1': composite_score_1,
        'Composite_Score_2': composite_score_2,
    }
    
    # Add individual cell line R² and Pearson values
    for cl in unique_cell_lines:
        if cl in cell_line_metrics:
            result_dict[f'R2_{cl}'] = cell_line_metrics[cl]['r2']
            result_dict[f'Pearson_{cl}'] = cell_line_metrics[cl]['pearson_r']
        else:
            result_dict[f'R2_{cl}'] = np.nan
            result_dict[f'Pearson_{cl}'] = np.nan
    
    results.append(result_dict)
    
    # Print summary
    print(f"\n📊 Overall Summary:")
    print(f"  Test R²: {test_r2:.4f} | Test Pearson: {test_pearson_r:.4f}")
    print(f"  R² Gap: {r2_gap:.4f} | Pearson Gap: {pearson_gap:.4f}")
    print(f"  Composite Score: {composite_score_2:.4f} | Time: {train_time:.1f}s")

# Create comprehensive results DataFrames
results_df = pd.DataFrame(results)
per_cell_line_df = pd.DataFrame(per_cell_line_results)

print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

# Display key metrics sorted by Composite Score 2
print("\n📋 OVERALL PERFORMANCE (Sorted by Composite Score):")
summary_cols = ['Configuration', 'Test_R2', 'Test_Pearson_R', 'R2_Gap', 
                'R2_Std_CellLines', 'Test_RMSE', 'Composite_Score_2']
results_summary = results_df[summary_cols].sort_values('Composite_Score_2', ascending=False)
print(results_summary.to_string(index=False))

# Display per-cell line consistency
print("\n📋 CELL LINE CONSISTENCY:")
consistency_cols = ['Configuration', 'R2_Mean_CellLines', 'R2_Std_CellLines', 
                    'R2_Min_CellLine', 'R2_Max_CellLine', 
                    'Pearson_Mean_CellLines', 'Pearson_Std_CellLines']
consistency_summary = results_df[consistency_cols].sort_values('R2_Std_CellLines')
print(consistency_summary.to_string(index=False))

# Find best performers for each criterion
print("\n" + "="*80)
print("🏆 BEST PERFORMERS BY CRITERION")
print("="*80)

best_test_r2 = results_df.loc[results_df['Test_R2'].idxmax()]
print(f"\n1. Highest Test R²: {best_test_r2['Configuration']}")
print(f"   Test R² = {best_test_r2['Test_R2']:.4f}")

best_pearson = results_df.loc[results_df['Test_Pearson_R'].idxmax()]
print(f"\n2. Highest Test Pearson: {best_pearson['Configuration']}")
print(f"   Test Pearson = {best_pearson['Test_Pearson_R']:.4f}")

best_generalization = results_df.loc[results_df['R2_Gap'].idxmin()]
print(f"\n3. Best Generalization (Lowest R² Gap): {best_generalization['Configuration']}")
print(f"   R² Gap = {best_generalization['R2_Gap']:.4f}")

best_consistency = results_df.loc[results_df['R2_Std_CellLines'].idxmin()]
print(f"\n4. Most Consistent Across Cell Lines: {best_consistency['Configuration']}")
print(f"   R² Std = {best_consistency['R2_Std_CellLines']:.4f}")

best_composite = results_df.loc[results_df['Composite_Score_2'].idxmax()]
print(f"\n5. Best Overall (Composite Score): {best_composite['Configuration']}")
print(f"   Composite = {best_composite['Composite_Score_2']:.4f}")

print("\n" + "="*80)

# ===== Section 4: Enhanced Comprehensive Visualizations =====

print("\n=== Creating Enhanced Comprehensive Visualizations ===")

# Create output directory for figures
FIGURES_DIR = "analysis/figures/bagging"
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Figures will be saved to: {FIGURES_DIR}/")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Shorten configuration names for display
results_df['Config_Short'] = results_df['Configuration'].str.replace('Config_', 'C').str.replace('_', '\n')

# ===== FIGURE 1: Overall Performance Dashboard (3x3 grid) =====
print("\n[1/4] Creating comprehensive dashboard...")
fig1, axes1 = plt.subplots(3, 3, figsize=(20, 16))

# --- Row 1: R² and Correlation Metrics ---

# 1.1 Test R² Comparison
ax = axes1[0, 0]
bars = ax.bar(results_df['Config_Short'], results_df['Test_R2'], 
              color=['#2ecc71' if i == results_df['Test_R2'].idxmax() else '#3498db' 
                     for i in results_df.index])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
ax.set_title('Test R² Performance', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 1.2 Test Pearson Correlation
ax = axes1[0, 1]
bars = ax.bar(results_df['Config_Short'], results_df['Test_Pearson_R'], 
              color=['#9b59b6' if i == results_df['Test_Pearson_R'].idxmax() else '#3498db' 
                     for i in results_df.index])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('Pearson Correlation', fontsize=11, fontweight='bold')
ax.set_title('Test Pearson R (Linear Correlation)', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 1.3 Test Spearman Correlation
ax = axes1[0, 2]
bars = ax.bar(results_df['Config_Short'], results_df['Test_Spearman_R'], 
              color=['#e67e22' if i == results_df['Test_Spearman_R'].idxmax() else '#3498db' 
                     for i in results_df.index])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('Spearman Correlation', fontsize=11, fontweight='bold')
ax.set_title('Test Spearman R (Rank Correlation)', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Row 2: Generalization Metrics ---

# 2.1 R² Gap (Overfitting)
ax = axes1[1, 0]
colors = ['#2ecc71' if gap < 0.15 else '#f39c12' if gap < 0.25 else '#e74c3c' 
          for gap in results_df['R2_Gap']]
bars = ax.bar(results_df['Config_Short'], results_df['R2_Gap'], color=colors)
ax.axhline(y=0.15, color='green', linestyle='--', lw=2, alpha=0.7, label='Good (<0.15)')
ax.axhline(y=0.25, color='orange', linestyle='--', lw=2, alpha=0.7, label='Moderate (<0.25)')
ax.set_ylabel('R² Gap (Train - Test)', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Overfitting Check (R² Gap)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2.2 Pearson Gap
ax = axes1[1, 1]
bars = ax.bar(results_df['Config_Short'], results_df['Pearson_Gap'], color='#95a5a6')
ax.axhline(y=0, color='red', linestyle='--', lw=2, alpha=0.7)
ax.set_ylabel('Pearson Gap (Train - Test)', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Correlation Generalization Gap', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2.3 Train vs Test Comparison
ax = axes1[1, 2]
x_pos = np.arange(len(results_df))
width = 0.35
ax.bar(x_pos - width/2, results_df['Train_R2'], width, 
       label='Train R²', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, results_df['Test_R2'], width, 
       label='Test R²', color='#e74c3c', alpha=0.8)
ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Train vs Test R²', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Config_Short'], fontsize=9)
ax.set_ylim([0, 1])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# --- Row 3: Cell Line Consistency & Efficiency ---

# 3.1 R² Consistency Across Cell Lines
ax = axes1[2, 0]
bars = ax.bar(results_df['Config_Short'], results_df['R2_Std_CellLines'], 
              color=['#2ecc71' if i == results_df['R2_Std_CellLines'].idxmin() else '#34495e' 
                     for i in results_df.index])
ax.set_ylabel('R² Std Dev', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Cell Line Consistency (Lower = More Consistent)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3.2 Test RMSE
ax = axes1[2, 1]
bars = ax.bar(results_df['Config_Short'], results_df['Test_RMSE'], color='#95a5a6')
ax.set_ylabel('Test RMSE', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Test RMSE (Lower is Better)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# 3.3 Composite Score 2
ax = axes1[2, 2]
bars = ax.bar(results_df['Config_Short'], results_df['Composite_Score_2'], 
              color=['#f1c40f' if i == results_df['Composite_Score_2'].idxmax() else '#34495e' 
                     for i in results_df.index])
ax.set_ylabel('Composite Score', fontsize=11, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Overall Composite Score (Higher is Better)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
figure1_path = os.path.join(FIGURES_DIR, '01_comprehensive_dashboard.png')
plt.savefig(figure1_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure1_path}")
plt.show()

# ===== FIGURE 2: Per-Cell Line Performance Heatmaps =====
print("\n[2/4] Creating cell line heatmaps...")
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))

# Prepare data for heatmaps
cell_line_cols_r2 = [col for col in results_df.columns if col.startswith('R2_') and col != 'R2_Gap' and col != 'R2_Ratio' and col != 'R2_Mean_CellLines' and col != 'R2_Std_CellLines' and col != 'R2_Min_CellLine' and col != 'R2_Max_CellLine']
cell_line_cols_pearson = [col for col in results_df.columns if col.startswith('Pearson_') and 'CellLines' not in col and 'Gap' not in col]

# 2.1 R² Heatmap
ax = axes2[0]
heatmap_data_r2 = results_df[['Configuration'] + cell_line_cols_r2].set_index('Configuration')
heatmap_data_r2.columns = [col.replace('R2_', '') for col in heatmap_data_r2.columns]
sns.heatmap(heatmap_data_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'R² Score'})
ax.set_title('R² Performance by Cell Line', fontsize=14, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')

# 2.2 Pearson Correlation Heatmap
ax = axes2[1]
heatmap_data_pearson = results_df[['Configuration'] + cell_line_cols_pearson].set_index('Configuration')
heatmap_data_pearson.columns = [col.replace('Pearson_', '') for col in heatmap_data_pearson.columns]
sns.heatmap(heatmap_data_pearson, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Pearson R'})
ax.set_title('Pearson Correlation by Cell Line', fontsize=14, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')

plt.tight_layout()
figure2_path = os.path.join(FIGURES_DIR, '02_cell_line_heatmaps.png')
plt.savefig(figure2_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure2_path}")
plt.show()

# ===== FIGURE 3: Cell Line Comparison (Box plots and Line plots) =====
print("\n[3/4] Creating cell line analysis...")
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 R² Distribution by Cell Line (Box Plot)
ax = axes3[0, 0]
per_cell_line_df_sorted = per_cell_line_df.sort_values('Cell_Line')
sns.boxplot(data=per_cell_line_df_sorted, x='Cell_Line', y='R2', ax=ax, palette='Set2')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_title('R² Distribution by Cell Line (All Configs)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 3.2 Pearson Distribution by Cell Line (Box Plot)
ax = axes3[0, 1]
sns.boxplot(data=per_cell_line_df_sorted, x='Cell_Line', y='Pearson_R', ax=ax, palette='Set3')
ax.set_ylabel('Pearson R', fontsize=12, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_title('Pearson R Distribution by Cell Line (All Configs)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 3.3 R² Across Configs for Each Cell Line (Line Plot) - FIXED
ax = axes3[1, 0]

# Create a mapping for configuration order (to ensure consistent x-axis)
config_order = {config: i for i, config in enumerate(results_df['Configuration'].tolist())}

# Add a column for config order
per_cell_line_df['config_order'] = per_cell_line_df['Configuration'].map(config_order)

# Plot each cell line
for cell_line in unique_cell_lines:
    cell_line_data = per_cell_line_df[per_cell_line_df['Cell_Line'] == cell_line].copy()
    
    # Sort by config order
    cell_line_data = cell_line_data.sort_values('config_order')
    
    # Plot
    ax.plot(cell_line_data['config_order'], 
            cell_line_data['R2'], 
            marker='o', 
            label=cell_line, 
            linewidth=2, 
            markersize=8)

ax.set_xlabel('Configuration Index', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('R² Trajectory by Cell Line', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(range(1, len(results_df)+1))
ax.legend(title='Cell Line', fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 0.65])  # Adjusted based on your data range

# 3.4 Mean vs Std R² Across Cell Lines (Scatter)
ax = axes3[1, 1]
scatter = ax.scatter(results_df['R2_Mean_CellLines'], 
                     results_df['R2_Std_CellLines'],
                     s=results_df['Composite_Score_2']*500,  # Size by composite score
                     c=results_df['Test_R2'], 
                     cmap='viridis',
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=1.5)

# Add configuration labels - shortened for clarity
for idx, row in results_df.iterrows():
    config_short_label = str(idx + 1)  # Just use numbers 1-7
    ax.annotate(config_short_label,
                (row['R2_Mean_CellLines'], row['R2_Std_CellLines']),
                fontsize=10, ha='center', fontweight='bold')

ax.set_xlabel('Mean R² Across Cell Lines', fontsize=12, fontweight='bold')
ax.set_ylabel('Std R² Across Cell Lines', fontsize=12, fontweight='bold')
ax.set_title('Consistency vs Performance\n(Size=Composite Score, Color=Test R²)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Test R²', fontsize=11, fontweight='bold')

plt.tight_layout()
figure3_path = os.path.join(FIGURES_DIR, '03_cell_line_analysis.png')
plt.savefig(figure3_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure3_path}")
plt.close()

# ===== FIGURE 4: Detailed Configuration Ranking (ORIGINAL VALUES) =====
print("\n[4/4] Creating multi-criteria ranking...")
fig4, ax4 = plt.subplots(figsize=(14, 8))

# Create ranking visualization
metrics_to_plot = ['Test_R2', 'Test_Pearson_R', 'Generalization_Score', 
                   'R2_Mean_CellLines', 'Composite_Score_2']
metric_names = ['Test R²', 'Test Pearson', 'Generalization', 'Cell Line\nMean R²', 'Composite']

# Use ORIGINAL values (no normalization)
original_data = results_df[metrics_to_plot].copy()

# Create grouped bar chart
x = np.arange(len(results_df))
width = 0.15
colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22', '#f1c40f']

for i, (metric, name, color) in enumerate(zip(metrics_to_plot, metric_names, colors)):
    offset = width * (i - 2)
    ax4.bar(x + offset, original_data[metric], width, label=name, color=color, alpha=0.8)

ax4.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax4.set_ylabel('Score (Original Values)', fontsize=12, fontweight='bold')
ax4.set_title('Multi-Criteria Performance Comparison (Original Values)', 
              fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([c.replace('Config_', 'C').replace('_', '\n') 
                      for c in results_df['Configuration']], fontsize=9, rotation=45, ha='right')
ax4.legend(loc='upper left', fontsize=10, ncol=5)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1.0])  # Since all metrics are in 0-1 range

plt.tight_layout()
figure4_path = os.path.join(FIGURES_DIR, '04_multi_criteria_ranking.png')
plt.savefig(figure4_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure4_path}")
plt.close()

print("\n" + "="*80)
print("✓ All visualizations created successfully!")
print("="*80)
print(f"\nAll figures saved to: {FIGURES_DIR}/")
print("\nGenerated files:")
print(f"  1. {os.path.join(FIGURES_DIR, '01_comprehensive_dashboard.png')}")
print(f"  2. {os.path.join(FIGURES_DIR, '02_cell_line_heatmaps.png')}")
print(f"  3. {os.path.join(FIGURES_DIR, '03_cell_line_analysis.png')}")
print(f"  4. {os.path.join(FIGURES_DIR, '04_multi_criteria_ranking.png')}")
print("="*80)

# ===== Section 5: Save Results and Model =====

print("\n" + "="*80)
print("SAVING RESULTS AND MODEL")
print("="*80)

# Create output directories
RESULTS_DIR = "models/results/bagging"
MODELS_DIR = "models/trained"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== 1. Save Results DataFrames =====
print("\n[1/5] Saving results dataframes...")

# Save comprehensive results
results_csv_path = os.path.join(RESULTS_DIR, "bagging_configurations_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"✓ Saved: {results_csv_path}")

# Save per-cell line results
per_cell_line_csv_path = os.path.join(RESULTS_DIR, "bagging_per_cell_line_results.csv")
per_cell_line_df.to_csv(per_cell_line_csv_path, index=False)
print(f"✓ Saved: {per_cell_line_csv_path}")

# ===== 2. Save Summary Report =====
print("\n[2/5] Saving summary report...")

report_path = os.path.join(RESULTS_DIR, "bagging_summary_report.txt")
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BAGGING MODEL CONFIGURATION ASSESSMENT REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"Dataset: {df.shape[0]} observations, {len(feature_cols)} features\n")
    f.write(f"Train set: {len(X_train)} observations ({len(train_genes)} genes)\n")
    f.write(f"Test set: {len(X_test)} observations ({len(test_genes)} genes)\n\n")
    
    f.write("="*80 + "\n")
    f.write("TOP PERFORMERS BY CRITERION\n")
    f.write("="*80 + "\n\n")
    
    best_test_r2 = results_df.loc[results_df['Test_R2'].idxmax()]
    f.write(f"1. Highest Test R²: {best_test_r2['Configuration']}\n")
    f.write(f"   Test R² = {best_test_r2['Test_R2']:.4f}\n\n")
    
    best_pearson = results_df.loc[results_df['Test_Pearson_R'].idxmax()]
    f.write(f"2. Highest Test Pearson: {best_pearson['Configuration']}\n")
    f.write(f"   Test Pearson = {best_pearson['Test_Pearson_R']:.4f}\n\n")
    
    best_generalization = results_df.loc[results_df['R2_Gap'].idxmin()]
    f.write(f"3. Best Generalization: {best_generalization['Configuration']}\n")
    f.write(f"   R² Gap = {best_generalization['R2_Gap']:.4f}\n\n")
    
    best_consistency = results_df.loc[results_df['R2_Std_CellLines'].idxmin()]
    f.write(f"4. Most Consistent: {best_consistency['Configuration']}\n")
    f.write(f"   R² Std = {best_consistency['R2_Std_CellLines']:.4f}\n\n")
    
    best_composite = results_df.loc[results_df['Composite_Score_2'].idxmax()]
    f.write(f"5. Best Overall: {best_composite['Configuration']}\n")
    f.write(f"   Composite = {best_composite['Composite_Score_2']:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("OVERALL PERFORMANCE SUMMARY (Sorted by Composite Score)\n")
    f.write("="*80 + "\n\n")
    f.write(results_summary.to_string(index=False))
    
    f.write("\n\n" + "="*80 + "\n")
    f.write("CELL LINE CONSISTENCY SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(consistency_summary.to_string(index=False))

print(f"✓ Saved: {report_path}")

# ===== 3. Save Best Model =====
print("\n[3/5] Training and saving best model...")

# Get best configuration
best_config_name = best_composite['Configuration']
best_params = param_configs[best_config_name]

print(f"Best configuration: {best_config_name}")
print(f"  Test R² = {best_composite['Test_R2']:.4f}")
print(f"  Composite Score = {best_composite['Composite_Score_2']:.4f}")

# Train best model
best_base_est = DecisionTreeRegressor(**best_params['base'], random_state=42)
best_model = BaggingRegressor(
    estimator=best_base_est,
    **best_params['bagging'],
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

print("Training best model...")
best_model.fit(X_train, y_train)

# Save model
model_path = os.path.join(MODELS_DIR, "best_bagging_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✓ Saved: {model_path}")

# Save model metadata
metadata = {
    'configuration_name': best_config_name,
    'parameters': best_params,
    'train_r2': best_composite['Train_R2'],
    'test_r2': best_composite['Test_R2'],
    'test_pearson': best_composite['Test_Pearson_R'],
    'r2_gap': best_composite['R2_Gap'],
    'composite_score': best_composite['Composite_Score_2'],
    'feature_names': feature_cols,
    'n_features': len(feature_cols),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'train_date': str(pd.Timestamp.now())
}

metadata_path = os.path.join(MODELS_DIR, "best_model_metadata.pkl")
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Saved: {metadata_path}")

# ===== 4. Save Train/Test Split Indices =====
print("\n[4/5] Saving train/test split indices...")

split_data = {
    'train_idx': train_idx,
    'test_idx': test_idx,
    'train_genes': list(train_genes),
    'test_genes': list(test_genes),
    'random_state': 42
}

split_path = os.path.join(RESULTS_DIR, "train_test_split.pkl")
with open(split_path, 'wb') as f:
    pickle.dump(split_data, f)

print(f"✓ Saved: {split_path}")

# ===== 5. Save Configuration Parameters =====
print("\n[5/5] Saving all configuration parameters...")

config_path = os.path.join(RESULTS_DIR, "all_configurations.pkl")
with open(config_path, 'wb') as f:
    pickle.dump(param_configs, f)

print(f"✓ Saved: {config_path}")

# ===== Summary =====
print("\n" + "="*80)
print("✓ ALL DATA SAVED SUCCESSFULLY!")
print("="*80)

print(f"\nResults saved to: {RESULTS_DIR}/")
print("  • bagging_configurations_results.csv - All configuration metrics")
print("  • bagging_per_cell_line_results.csv - Cell line specific performance")
print("  • bagging_summary_report.txt - Human-readable summary")
print("  • train_test_split.pkl - Train/test indices for reproducibility")
print("  • all_configurations.pkl - All tested parameter configurations")

print(f"\nModel saved to: {MODELS_DIR}/")
print("  • best_bagging_model.pkl - Trained best model")
print("  • best_model_metadata.pkl - Model metadata and parameters")

print(f"\nFigures saved to: {FIGURES_DIR}/")
print("  • 01_comprehensive_dashboard.png")
print("  • 02_cell_line_heatmaps.png")
print("  • 03_cell_line_analysis.png")
print("  • 04_multi_criteria_ranking.png")

print("\n" + "="*80)
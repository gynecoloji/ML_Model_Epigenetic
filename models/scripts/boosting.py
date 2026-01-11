import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
from sklearn.model_selection import GroupKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit

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

print("="*80)
print("BOOSTING WITH 5-FOLD CROSS-VALIDATION")
print("="*80)

print(f"\nData split:")
print(f"  Training set: {len(X_train)} observations ({len(set(groups_train))} genes)")
print(f"  Test set: {len(X_test)} observations ({len(set(groups_test))} genes)")
print(f"  CV Strategy: 5-Fold Gene-Based Cross-Validation on training set")

# ===== Define Boosting Configurations =====

boosting_configs = {
    # Gradient Boosting (sklearn)
    'GradientBoosting_Conservative': {
        'model_type': 'GradientBoosting',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 3,
            'min_samples_split': 50,
            'min_samples_leaf': 25,
            'subsample': 0.8,
            'max_features': 0.7,
            'random_state': 42
        }
    },
    'GradientBoosting_Moderate': {
        'model_type': 'GradientBoosting',
        'params': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_samples_split': 30,
            'min_samples_leaf': 15,
            'subsample': 0.8,
            'max_features': 0.8,
            'random_state': 42
        }
    },
    'GradientBoosting_Aggressive': {
        'model_type': 'GradientBoosting',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'max_features': 0.9,
            'random_state': 42
        }
    },
    
    # XGBoost
    'XGBoost_Conservative': {
        'model_type': 'XGBoost',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 3,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'XGBoost_Moderate': {
        'model_type': 'XGBoost',
        'params': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'XGBoost_Aggressive': {
        'model_type': 'XGBoost',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.01,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    
    # LightGBM
    'LightGBM_Conservative': {
        'model_type': 'LightGBM',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 3,
            'num_leaves': 15,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    },
    'LightGBM_Moderate': {
        'model_type': 'LightGBM',
        'params': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 4,
            'num_leaves': 31,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    },
    'LightGBM_Aggressive': {
        'model_type': 'LightGBM',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'num_leaves': 50,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.01,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    },
}

print(f"\nTesting {len(boosting_configs)} boosting configurations:")
for config_name in boosting_configs.keys():
    print(f"  • {config_name}")

# ===== Setup 5-Fold Cross-Validation =====

# Use GroupKFold to ensure genes stay together
n_folds = 5
group_kfold = GroupKFold(n_splits=n_folds)

print(f"\n5-Fold Cross-Validation Setup:")
print(f"  • {n_folds} folds")
print(f"  • Gene-based splitting (GroupKFold)")
print(f"  • Each configuration will be trained {n_folds} times")

# Store results
cv_results = []
final_test_results = []
per_cell_line_results = []

# Get cell line info for test set
test_cell_lines = df.iloc[test_idx]['cell_line'].values
unique_cell_lines = sorted(df['cell_line'].unique())

# ===== Test Each Configuration with Cross-Validation =====

for config_name, config in boosting_configs.items():
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    
    model_type = config['model_type']
    params = config['params']
    
    print(f"Model Type: {model_type}")
    print(f"Key params: n_estimators={params['n_estimators']}, "
          f"learning_rate={params['learning_rate']}, "
          f"max_depth={params['max_depth']}")
    
    # ===== CROSS-VALIDATION ON TRAINING SET =====
    print(f"\n🔄 Running 5-Fold Cross-Validation...")
    
    cv_scores = {
        'r2': [],
        'rmse': [],
        'mae': [],
        'pearson': [],
        'spearman': []
    }
    
    fold_num = 1
    cv_start_time = time.time()
    
    for train_fold_idx, val_fold_idx in group_kfold.split(X_train, y_train, groups=groups_train):
        print(f"  Fold {fold_num}/{n_folds}...", end=" ")
        
        # Split into train and validation for this fold
        X_train_fold = X_train[train_fold_idx]
        y_train_fold = y_train[train_fold_idx]
        X_val_fold = X_train[val_fold_idx]
        y_val_fold = y_train[val_fold_idx]
        
        # Create and train model
        if model_type == 'GradientBoosting':
            model_fold = GradientBoostingRegressor(**params)
        elif model_type == 'XGBoost':
            model_fold = XGBRegressor(**params)
        elif model_type == 'LightGBM':
            model_fold = LGBMRegressor(**params)
        
        model_fold.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_val_pred = model_fold.predict(X_val_fold)
        
        # Calculate metrics
        fold_r2 = r2_score(y_val_fold, y_val_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        fold_mae = mean_absolute_error(y_val_fold, y_val_pred)
        fold_pearson, _ = pearsonr(y_val_fold, y_val_pred)
        fold_spearman, _ = spearmanr(y_val_fold, y_val_pred)
        
        cv_scores['r2'].append(fold_r2)
        cv_scores['rmse'].append(fold_rmse)
        cv_scores['mae'].append(fold_mae)
        cv_scores['pearson'].append(fold_pearson)
        cv_scores['spearman'].append(fold_spearman)
        
        print(f"R²={fold_r2:.4f}")
        fold_num += 1
    
    cv_time = time.time() - cv_start_time
    
    # Calculate CV statistics
    cv_r2_mean = np.mean(cv_scores['r2'])
    cv_r2_std = np.std(cv_scores['r2'])
    cv_rmse_mean = np.mean(cv_scores['rmse'])
    cv_rmse_std = np.std(cv_scores['rmse'])
    cv_mae_mean = np.mean(cv_scores['mae'])
    cv_pearson_mean = np.mean(cv_scores['pearson'])
    cv_pearson_std = np.std(cv_scores['pearson'])
    cv_spearman_mean = np.mean(cv_scores['spearman'])
    
    print(f"\n📊 Cross-Validation Results (mean ± std):")
    print(f"  R²:       {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"  RMSE:     {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
    print(f"  MAE:      {cv_mae_mean:.4f}")
    print(f"  Pearson:  {cv_pearson_mean:.4f} ± {cv_pearson_std:.4f}")
    print(f"  Spearman: {cv_spearman_mean:.4f}")
    print(f"  CV Time:  {cv_time:.1f}s")
    
    # ===== TRAIN FINAL MODEL ON FULL TRAINING SET =====
    print(f"\n🎯 Training final model on full training set...")
    
    if model_type == 'GradientBoosting':
        final_model = GradientBoostingRegressor(**params)
    elif model_type == 'XGBoost':
        final_model = XGBRegressor(**params)
    elif model_type == 'LightGBM':
        final_model = LGBMRegressor(**params)
    
    train_start_time = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - train_start_time
    
    # ===== EVALUATE ON HELD-OUT TEST SET =====
    print(f"📈 Evaluating on held-out test set...")
    
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)
    
    # Overall performance
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Correlations
    train_pearson_r, _ = pearsonr(y_train, y_train_pred)
    test_pearson_r, _ = pearsonr(y_test, y_test_pred)
    train_spearman_r, _ = spearmanr(y_train, y_train_pred)
    test_spearman_r, _ = spearmanr(y_test, y_test_pred)
    
    # Generalization metrics
    r2_gap = train_r2 - test_r2
    r2_ratio = test_r2 / train_r2 if train_r2 > 0 else 0
    rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')
    pearson_gap = train_pearson_r - test_pearson_r
    generalization_score = 1 - abs(r2_gap) if abs(r2_gap) < 1 else 0
    
    # Residuals
    residuals_test = y_test - y_test_pred
    residuals_std = np.std(residuals_test)
    residuals_mean_abs = np.mean(np.abs(residuals_test))
    
    # ===== Per-Cell Line Assessment =====
    cell_line_metrics = {}
    
    print(f"\n📊 Per-Cell Line Performance (Test Set):")
    print(f"{'Cell Line':<12} {'N':<6} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'Pearson':<10}")
    print("-" * 60)
    
    for cell_line in unique_cell_lines:
        mask = test_cell_lines == cell_line
        n_samples = mask.sum()
        
        if n_samples > 0:
            cl_y_true = y_test[mask]
            cl_y_pred = y_test_pred[mask]
            
            cl_r2 = r2_score(cl_y_true, cl_y_pred)
            cl_rmse = np.sqrt(mean_squared_error(cl_y_true, cl_y_pred))
            cl_mae = mean_absolute_error(cl_y_true, cl_y_pred)
            cl_pearson_r, _ = pearsonr(cl_y_true, cl_y_pred)
            cl_spearman_r, _ = spearmanr(cl_y_true, cl_y_pred)
            
            cell_line_metrics[cell_line] = {
                'n_samples': n_samples,
                'r2': cl_r2,
                'rmse': cl_rmse,
                'mae': cl_mae,
                'pearson_r': cl_pearson_r,
                'spearman_r': cl_spearman_r
            }
            
            print(f"{cell_line:<12} {n_samples:<6} {cl_r2:<8.4f} {cl_rmse:<8.4f} "
                  f"{cl_mae:<8.4f} {cl_pearson_r:<10.4f}")
            
            per_cell_line_results.append({
                'Configuration': config_name,
                'Model_Type': model_type,
                'Cell_Line': cell_line,
                'N_Samples': n_samples,
                'R2': cl_r2,
                'RMSE': cl_rmse,
                'MAE': cl_mae,
                'Pearson_R': cl_pearson_r,
                'Spearman_R': cl_spearman_r
            })
    
    # Cell line consistency
    r2_values = [m['r2'] for m in cell_line_metrics.values()]
    pearson_values = [m['pearson_r'] for m in cell_line_metrics.values()]
    
    r2_mean_across_lines = np.mean(r2_values)
    r2_std_across_lines = np.std(r2_values)
    r2_min_across_lines = np.min(r2_values)
    r2_max_across_lines = np.max(r2_values)
    
    pearson_mean_across_lines = np.mean(pearson_values)
    pearson_std_across_lines = np.std(pearson_values)
    pearson_min_across_lines = np.min(pearson_values)
    
    # ===== Composite Scores =====
    composite_score_1 = test_r2 - (r2_gap * 0.5)
    
    norm_test_r2 = test_r2
    norm_test_pearson = (test_pearson_r + 1) / 2
    norm_gap = max(0, 1 - (r2_gap / 0.5))
    norm_consistency = max(0, 1 - (r2_std_across_lines / 0.3))
    norm_efficiency = max(0, 1 - (train_time / 300))
    
    composite_score_2 = (0.35 * norm_test_r2 + 
                         0.25 * norm_test_pearson +
                         0.20 * norm_gap + 
                         0.15 * norm_consistency + 
                         0.05 * norm_efficiency)
    
    # ===== Store Results =====
    result_dict = {
        'Configuration': config_name,
        'Model_Type': model_type,
        # Cross-Validation Results
        'CV_R2_Mean': cv_r2_mean,
        'CV_R2_Std': cv_r2_std,
        'CV_RMSE_Mean': cv_rmse_mean,
        'CV_RMSE_Std': cv_rmse_std,
        'CV_MAE_Mean': cv_mae_mean,
        'CV_Pearson_Mean': cv_pearson_mean,
        'CV_Pearson_Std': cv_pearson_std,
        'CV_Spearman_Mean': cv_spearman_mean,
        'CV_Time_sec': cv_time,
        # Final Model Performance
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
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
        # Composite
        'Composite_Score_1': composite_score_1,
        'Composite_Score_2': composite_score_2,
    }
    
    # Add individual cell line metrics
    for cl in unique_cell_lines:
        if cl in cell_line_metrics:
            result_dict[f'R2_{cl}'] = cell_line_metrics[cl]['r2']
            result_dict[f'Pearson_{cl}'] = cell_line_metrics[cl]['pearson_r']
        else:
            result_dict[f'R2_{cl}'] = np.nan
            result_dict[f'Pearson_{cl}'] = np.nan
    
    cv_results.append(result_dict)
    
    # Print final summary
    print(f"\n✅ Final Test Set Performance:")
    print(f"  Test R²:      {test_r2:.4f} (CV: {cv_r2_mean:.4f} ± {cv_r2_std:.4f})")
    print(f"  Test Pearson: {test_pearson_r:.4f} (CV: {cv_pearson_mean:.4f} ± {cv_pearson_std:.4f})")
    print(f"  R² Gap:       {r2_gap:.4f}")
    print(f"  Composite:    {composite_score_2:.4f}")
    print(f"  Total Time:   {cv_time + train_time:.1f}s")

# Create results DataFrames
results_df = pd.DataFrame(cv_results)
per_cell_line_df = pd.DataFrame(per_cell_line_results)

print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

# ===== Section 3: Results Summary & Ranking =====

print("\n" + "="*80)
print("BOOSTING RESULTS SUMMARY")
print("="*80)

# ===== 1. Overall Performance Summary =====
print("\n📋 OVERALL PERFORMANCE (Sorted by Test R²):")
summary_cols = ['Configuration', 'Model_Type', 'CV_R2_Mean', 'CV_R2_Std', 
                'Test_R2', 'R2_Gap', 'Test_Pearson_R', 'Composite_Score_2']
results_summary = results_df[summary_cols].sort_values('Test_R2', ascending=False)
print(results_summary.to_string(index=False))

# ===== 2. Cross-Validation vs Test Performance =====
print("\n📋 CV vs TEST PERFORMANCE:")
cv_test_cols = ['Configuration', 'Model_Type', 'CV_R2_Mean', 'Test_R2', 
                'CV_Pearson_Mean', 'Test_Pearson_R']
cv_test_summary = results_df[cv_test_cols].sort_values('Test_R2', ascending=False)
print(cv_test_summary.to_string(index=False))

# ===== 3. Model Type Comparison =====
print("\n📋 PERFORMANCE BY MODEL TYPE:")
model_type_summary = results_df.groupby('Model_Type').agg({
    'Test_R2': ['mean', 'std', 'max'],
    'Test_Pearson_R': ['mean', 'std', 'max'],
    'R2_Gap': ['mean', 'std', 'min'],
    'Train_Time_sec': ['mean', 'sum']
}).round(4)
print(model_type_summary)

# ===== 4. Cell Line Consistency =====
print("\n📋 CELL LINE CONSISTENCY:")
consistency_cols = ['Configuration', 'Model_Type', 'R2_Mean_CellLines', 
                    'R2_Std_CellLines', 'R2_Min_CellLine', 'R2_Max_CellLine']
consistency_summary = results_df[consistency_cols].sort_values('R2_Std_CellLines')
print(consistency_summary.to_string(index=False))

# ===== 5. Find Best Performers =====
print("\n" + "="*80)
print("🏆 BEST PERFORMERS BY CRITERION")
print("="*80)

best_test_r2 = results_df.loc[results_df['Test_R2'].idxmax()]
print(f"\n1. Highest Test R²: {best_test_r2['Configuration']}")
print(f"   Model Type: {best_test_r2['Model_Type']}")
print(f"   Test R² = {best_test_r2['Test_R2']:.4f}")
print(f"   CV R² = {best_test_r2['CV_R2_Mean']:.4f} ± {best_test_r2['CV_R2_Std']:.4f}")

best_cv_r2 = results_df.loc[results_df['CV_R2_Mean'].idxmax()]
print(f"\n2. Highest CV R²: {best_cv_r2['Configuration']}")
print(f"   Model Type: {best_cv_r2['Model_Type']}")
print(f"   CV R² = {best_cv_r2['CV_R2_Mean']:.4f} ± {best_cv_r2['CV_R2_Std']:.4f}")
print(f"   Test R² = {best_cv_r2['Test_R2']:.4f}")

best_pearson = results_df.loc[results_df['Test_Pearson_R'].idxmax()]
print(f"\n3. Highest Test Pearson: {best_pearson['Configuration']}")
print(f"   Model Type: {best_pearson['Model_Type']}")
print(f"   Test Pearson = {best_pearson['Test_Pearson_R']:.4f}")

best_generalization = results_df.loc[results_df['R2_Gap'].idxmin()]
print(f"\n4. Best Generalization (Lowest R² Gap): {best_generalization['Configuration']}")
print(f"   Model Type: {best_generalization['Model_Type']}")
print(f"   R² Gap = {best_generalization['R2_Gap']:.4f}")

best_consistency = results_df.loc[results_df['R2_Std_CellLines'].idxmin()]
print(f"\n5. Most Consistent Across Cell Lines: {best_consistency['Configuration']}")
print(f"   Model Type: {best_consistency['Model_Type']}")
print(f"   R² Std = {best_consistency['R2_Std_CellLines']:.4f}")

best_composite = results_df.loc[results_df['Composite_Score_2'].idxmax()]
print(f"\n6. Best Overall (Composite Score): {best_composite['Configuration']}")
print(f"   Model Type: {best_composite['Model_Type']}")
print(f"   Composite = {best_composite['Composite_Score_2']:.4f}")
print(f"   Test R² = {best_composite['Test_R2']:.4f}")

# ===== 6. CV Reliability Analysis =====
print("\n" + "="*80)
print("📊 CV RELIABILITY ANALYSIS")
print("="*80)

results_df['CV_Test_R2_Diff'] = abs(results_df['CV_R2_Mean'] - results_df['Test_R2'])
results_df['CV_Test_Agreement'] = results_df['Test_R2'] / results_df['CV_R2_Mean']

print("\nCV vs Test Agreement (sorted by difference):")
reliability_cols = ['Configuration', 'CV_R2_Mean', 'Test_R2', 'CV_Test_R2_Diff', 'CV_Test_Agreement']
reliability_summary = results_df[reliability_cols].sort_values('CV_Test_R2_Diff')
print(reliability_summary.to_string(index=False))

print(f"\nMean CV-Test R² difference: {results_df['CV_Test_R2_Diff'].mean():.4f}")
print(f"Max CV-Test R² difference: {results_df['CV_Test_R2_Diff'].max():.4f}")

print("\n" + "="*80)

# ===== Section 4: Comprehensive Visualizations =====

print("\n=== Creating Comprehensive Visualizations ===")

# Create output directory
FIGURES_DIR = "analysis/figures/boosting"
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Figures will be saved to: {FIGURES_DIR}/")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Shorten configuration names for display
results_df['Config_Short'] = results_df['Configuration'].str.replace('GradientBoosting', 'GB').str.replace('XGBoost', 'XGB').str.replace('LightGBM', 'LGBM').str.replace('_Conservative', '\nCons').str.replace('_Moderate', '\nMod').str.replace('_Aggressive', '\nAgg')

# ===== FIGURE 1: Performance Comparison Dashboard (3x3 grid) =====
print("\n[1/5] Creating performance dashboard...")
fig1, axes1 = plt.subplots(3, 3, figsize=(20, 16))

# Define color mapping for model types
model_colors = {'GradientBoosting': '#3498db', 'XGBoost': '#e74c3c', 'LightGBM': '#2ecc71'}
colors = [model_colors[mt] for mt in results_df['Model_Type']]

# --- Row 1: R² Metrics ---

# 1.1 Test R² Comparison
ax = axes1[0, 0]
bars = ax.bar(results_df['Config_Short'], results_df['Test_R2'], color=colors, alpha=0.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
ax.set_title('Test R² Performance', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 1.2 CV R² (with error bars)
ax = axes1[0, 1]
ax.bar(results_df['Config_Short'], results_df['CV_R2_Mean'], 
       yerr=results_df['CV_R2_Std'], color=colors, alpha=0.8, capsize=5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('CV R² Score', fontsize=11, fontweight='bold')
ax.set_title('Cross-Validation R² (Mean ± Std)', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 1.3 CV vs Test R² Scatter
ax = axes1[0, 2]

for model_type, color in model_colors.items():
    mask = results_df['Model_Type'] == model_type
    ax.scatter(results_df.loc[mask, 'CV_R2_Mean'], 
               results_df.loc[mask, 'Test_R2'],
               label=model_type, color=color, s=150, alpha=0.7, 
               edgecolors='black', linewidth=1.5)

cv_min = results_df['CV_R2_Mean'].min()
cv_max = results_df['CV_R2_Mean'].max()
test_min = results_df['Test_R2'].min()
test_max = results_df['Test_R2'].max()

overall_min = min(cv_min, test_min) - 0.05
overall_max = max(cv_max, test_max) + 0.05

ax.set_xlim([overall_min, overall_max])
ax.set_ylim([overall_min, overall_max])

ax.plot([overall_min, overall_max], [overall_min, overall_max], 
        'k--', alpha=0.5, lw=2, label='Perfect Agreement', zorder=1)

ax.set_xlabel('CV R² Mean', fontsize=11, fontweight='bold')
ax.set_ylabel('Test R²', fontsize=11, fontweight='bold')
ax.set_title('CV vs Test R² Agreement', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

ax.set_aspect('equal', adjustable='box')

# --- Row 2: Correlation & Generalization ---

# 2.1 Test Pearson Correlation
ax = axes1[1, 0]
bars = ax.bar(results_df['Config_Short'], results_df['Test_Pearson_R'], color=colors, alpha=0.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('Pearson Correlation', fontsize=11, fontweight='bold')
ax.set_title('Test Pearson R', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 2.2 R² Gap (Overfitting)
ax = axes1[1, 1]
gap_colors = ['#2ecc71' if gap < 0.15 else '#f39c12' if gap < 0.25 else '#e74c3c' 
              for gap in results_df['R2_Gap']]
bars = ax.bar(results_df['Config_Short'], results_df['R2_Gap'], color=gap_colors, alpha=0.8)
ax.axhline(y=0.15, color='green', linestyle='--', lw=2, alpha=0.7, label='Good (<0.15)')
ax.axhline(y=0.25, color='orange', linestyle='--', lw=2, alpha=0.7, label='Moderate (<0.25)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel('R² Gap (Train - Test)', fontsize=11, fontweight='bold')
ax.set_title('Overfitting Check', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 2.3 Train vs Test R²
ax = axes1[1, 2]
x_pos = np.arange(len(results_df))
width = 0.35
ax.bar(x_pos - width/2, results_df['Train_R2'], width, 
       label='Train R²', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, results_df['Test_R2'], width, 
       label='Test R²', color='#e74c3c', alpha=0.8)
ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax.set_title('Train vs Test R²', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Config_Short'], fontsize=8, rotation=45, ha='right')
ax.set_ylim([0, 1])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# --- Row 3: Cell Line Consistency & Composite ---

# 3.1 Cell Line Consistency
ax = axes1[2, 0]
bars = ax.bar(results_df['Config_Short'], results_df['R2_Std_CellLines'], 
              color=colors, alpha=0.8)
ax.set_ylabel('R² Std Dev', fontsize=11, fontweight='bold')
ax.set_title('Cell Line Consistency (Lower = Better)', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 3.2 Test RMSE
ax = axes1[2, 1]
bars = ax.bar(results_df['Config_Short'], results_df['Test_RMSE'], color=colors, alpha=0.8)
ax.set_ylabel('Test RMSE', fontsize=11, fontweight='bold')
ax.set_title('Test RMSE (Lower = Better)', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 3.3 Composite Score
ax = axes1[2, 2]
bars = ax.bar(results_df['Config_Short'], results_df['Composite_Score_2'], 
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Composite Score', fontsize=11, fontweight='bold')
ax.set_title('Overall Composite Score (Higher = Better)', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
figure1_path = os.path.join(FIGURES_DIR, '01_performance_dashboard.png')
plt.savefig(figure1_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure1_path}")
plt.close()

# ===== FIGURE 2: Model Type Comparison =====
print("\n[2/5] Creating model type comparison...")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

# 2.1 Box plot - Test R² by Model Type
ax = axes2[0, 0]
model_order = ['GradientBoosting', 'XGBoost', 'LightGBM']
sns.boxplot(data=results_df, x='Model_Type', y='Test_R2', ax=ax, 
            order=model_order, palette=model_colors)
ax.set_ylabel('Test R²', fontsize=12, fontweight='bold')
ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
ax.set_title('Test R² Distribution by Model Type', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 2.2 Box plot - R² Gap by Model Type
ax = axes2[0, 1]
sns.boxplot(data=results_df, x='Model_Type', y='R2_Gap', ax=ax, 
            order=model_order, palette=model_colors)
ax.axhline(y=0.15, color='green', linestyle='--', lw=2, alpha=0.5)
ax.axhline(y=0.25, color='orange', linestyle='--', lw=2, alpha=0.5)
ax.set_ylabel('R² Gap', fontsize=12, fontweight='bold')
ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
ax.set_title('Overfitting by Model Type', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 2.3 Training Time by Model Type
ax = axes2[1, 0]
model_times = results_df.groupby('Model_Type')['Train_Time_sec'].mean()
bars = ax.bar(model_times.index, model_times.values, 
              color=[model_colors[mt] for mt in model_times.index], alpha=0.8)
ax.set_ylabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
ax.set_title('Training Efficiency by Model Type', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2.4 Performance Summary Table
ax = axes2[1, 1]
ax.axis('tight')
ax.axis('off')

summary_data = []
for model_type in model_order:
    mask = results_df['Model_Type'] == model_type
    summary_data.append([
        model_type,
        f"{results_df.loc[mask, 'Test_R2'].mean():.4f}",
        f"{results_df.loc[mask, 'Test_R2'].max():.4f}",
        f"{results_df.loc[mask, 'R2_Gap'].mean():.4f}",
        f"{results_df.loc[mask, 'Train_Time_sec'].mean():.1f}s"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Model Type', 'Mean R²', 'Max R²', 'Mean Gap', 'Avg Time'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows by model type
for i, model_type in enumerate(model_order, 1):
    for j in range(5):
        table[(i, j)].set_facecolor(model_colors[model_type])
        table[(i, j)].set_alpha(0.3)

ax.set_title('Performance Summary by Model Type', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
figure2_path = os.path.join(FIGURES_DIR, '02_model_type_comparison.png')
plt.savefig(figure2_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure2_path}")
plt.close()

# ===== FIGURE 3: Cell Line Heatmaps =====
print("\n[3/5] Creating cell line heatmaps...")
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 6))

# Prepare data for heatmaps
cell_line_cols_r2 = [col for col in results_df.columns 
                     if col.startswith('R2_') 
                     and col not in ['R2_Gap', 'R2_Ratio', 'R2_Mean_CellLines', 
                                     'R2_Std_CellLines', 'R2_Min_CellLine', 'R2_Max_CellLine']]
cell_line_cols_pearson = [col for col in results_df.columns 
                          if col.startswith('Pearson_') 
                          and 'CellLines' not in col and 'Gap' not in col]

# 3.1 R² Heatmap
ax = axes3[0]
heatmap_data_r2 = results_df[['Configuration'] + cell_line_cols_r2].set_index('Configuration')
heatmap_data_r2.columns = [col.replace('R2_', '') for col in heatmap_data_r2.columns]
sns.heatmap(heatmap_data_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.4, vmax=0.7, ax=ax, cbar_kws={'label': 'R² Score'})
ax.set_title('R² Performance by Cell Line', fontsize=14, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')

# 3.2 Pearson Correlation Heatmap
ax = axes3[1]
heatmap_data_pearson = results_df[['Configuration'] + cell_line_cols_pearson].set_index('Configuration')
heatmap_data_pearson.columns = [col.replace('Pearson_', '') for col in heatmap_data_pearson.columns]
sns.heatmap(heatmap_data_pearson, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.6, vmax=0.8, ax=ax, cbar_kws={'label': 'Pearson R'})
ax.set_title('Pearson Correlation by Cell Line', fontsize=14, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')

plt.tight_layout()
figure3_path = os.path.join(FIGURES_DIR, '03_cell_line_heatmaps.png')
plt.savefig(figure3_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure3_path}")
plt.close()

# ===== FIGURE 4: Cell Line Analysis =====
print("\n[4/5] Creating cell line analysis...")
fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))

# 4.1 R² Distribution by Cell Line
ax = axes4[0, 0]
per_cell_line_df_sorted = per_cell_line_df.sort_values('Cell_Line')
sns.boxplot(data=per_cell_line_df_sorted, x='Cell_Line', y='R2', ax=ax, palette='Set2')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_title('R² Distribution by Cell Line (All Configs)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 4.2 Pearson Distribution by Cell Line
ax = axes4[0, 1]
sns.boxplot(data=per_cell_line_df_sorted, x='Cell_Line', y='Pearson_R', ax=ax, palette='Set3')
ax.set_ylabel('Pearson R', fontsize=12, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_title('Pearson R Distribution by Cell Line (All Configs)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 4.3 R² by Cell Line and Model Type
ax = axes4[1, 0]
sns.boxplot(data=per_cell_line_df_sorted, x='Cell_Line', y='R2', hue='Model_Type', 
            ax=ax, palette=model_colors)
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Cell Line', fontsize=12, fontweight='bold')
ax.set_title('R² by Cell Line and Model Type', fontsize=13, fontweight='bold')
ax.legend(title='Model Type', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 4.4 Mean vs Std R² Scatter
ax = axes4[1, 1]
for model_type, color in model_colors.items():
    mask = results_df['Model_Type'] == model_type
    ax.scatter(results_df.loc[mask, 'R2_Mean_CellLines'], 
               results_df.loc[mask, 'R2_Std_CellLines'],
               s=results_df.loc[mask, 'Composite_Score_2']*500,
               c=color, label=model_type, alpha=0.7, 
               edgecolors='black', linewidth=1.5)

ax.set_xlabel('Mean R² Across Cell Lines', fontsize=12, fontweight='bold')
ax.set_ylabel('Std R² Across Cell Lines', fontsize=12, fontweight='bold')
ax.set_title('Consistency vs Performance\n(Size = Composite Score)', 
             fontsize=13, fontweight='bold')
ax.legend(title='Model Type', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
figure4_path = os.path.join(FIGURES_DIR, '04_cell_line_analysis.png')
plt.savefig(figure4_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure4_path}")
plt.close()

# ===== FIGURE 5: Multi-Criteria Ranking =====
print("\n[5/5] Creating multi-criteria ranking...")
fig5, ax5 = plt.subplots(figsize=(14, 8))

metrics_to_plot = ['Test_R2', 'Test_Pearson_R', 'Generalization_Score', 
                   'R2_Mean_CellLines', 'Composite_Score_2']
metric_names = ['Test R²', 'Test Pearson', 'Generalization', 'Cell Line\nMean R²', 'Composite']

# Use original values
original_data = results_df[metrics_to_plot].copy()

# Create grouped bar chart
x = np.arange(len(results_df))
width = 0.15
bar_colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22', '#f1c40f']

for i, (metric, name, color) in enumerate(zip(metrics_to_plot, metric_names, bar_colors)):
    offset = width * (i - 2)
    ax5.bar(x + offset, original_data[metric], width, label=name, color=color, alpha=0.8)

ax5.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax5.set_ylabel('Score (Original Values)', fontsize=12, fontweight='bold')
ax5.set_title('Multi-Criteria Performance Comparison', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(results_df['Config_Short'], fontsize=9, rotation=45, ha='right')
ax5.legend(loc='upper left', fontsize=10, ncol=5)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 1.0])

plt.tight_layout()
figure5_path = os.path.join(FIGURES_DIR, '05_multi_criteria_ranking.png')
plt.savefig(figure5_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figure5_path}")
plt.close()

print("\n" + "="*80)
print("✓ All visualizations created successfully!")
print("="*80)
print(f"\nAll figures saved to: {FIGURES_DIR}/")
print("\nGenerated files:")
print(f"  1. {os.path.join(FIGURES_DIR, '01_performance_dashboard.png')}")
print(f"  2. {os.path.join(FIGURES_DIR, '02_model_type_comparison.png')}")
print(f"  3. {os.path.join(FIGURES_DIR, '03_cell_line_heatmaps.png')}")
print(f"  4. {os.path.join(FIGURES_DIR, '04_cell_line_analysis.png')}")
print(f"  5. {os.path.join(FIGURES_DIR, '05_multi_criteria_ranking.png')}")
print("="*80)

# ===== Section 5: Save Results and Models =====

print("\n" + "="*80)
print("SAVING RESULTS AND MODELS")
print("="*80)

# Create output directories
RESULTS_DIR = "models/results/boosting"
MODELS_DIR = "models/trained"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== 1. Save Results DataFrames =====
print("\n[1/6] Saving results dataframes...")

# Save comprehensive results with CV scores
results_csv_path = os.path.join(RESULTS_DIR, "boosting_configurations_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"✓ Saved: {results_csv_path}")

# Save per-cell line results
per_cell_line_csv_path = os.path.join(RESULTS_DIR, "boosting_per_cell_line_results.csv")
per_cell_line_df.to_csv(per_cell_line_csv_path, index=False)
print(f"✓ Saved: {per_cell_line_csv_path}")

# ===== 2. Save Model Type Summary =====
print("\n[2/6] Saving model type comparison...")

model_type_comparison = results_df.groupby('Model_Type').agg({
    'Test_R2': ['mean', 'std', 'min', 'max'],
    'CV_R2_Mean': ['mean', 'std'],
    'Test_Pearson_R': ['mean', 'std', 'max'],
    'R2_Gap': ['mean', 'std', 'min'],
    'R2_Std_CellLines': ['mean', 'std', 'min'],
    'Train_Time_sec': ['mean', 'sum'],
    'Composite_Score_2': ['mean', 'max']
}).round(4)

model_comparison_path = os.path.join(RESULTS_DIR, "model_type_comparison.csv")
model_type_comparison.to_csv(model_comparison_path)
print(f"✓ Saved: {model_comparison_path}")

# ===== 3. Save Summary Report =====
print("\n[3/6] Saving comprehensive summary report...")

report_path = os.path.join(RESULTS_DIR, "boosting_summary_report.txt")
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BOOSTING MODEL ASSESSMENT REPORT WITH 5-FOLD CROSS-VALIDATION\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"Dataset: {df.shape[0]} observations, {len(feature_cols)} features\n")
    f.write(f"Train set: {len(X_train)} observations ({len(set(groups_train))} genes)\n")
    f.write(f"Test set:  {len(X_test)} observations ({len(set(groups_test))} genes)\n")
    f.write(f"CV Strategy: 5-Fold Gene-Based Cross-Validation\n\n")
    
    f.write("="*80 + "\n")
    f.write("MODEL TYPES TESTED\n")
    f.write("="*80 + "\n\n")
    f.write("1. Gradient Boosting (sklearn) - 3 configurations\n")
    f.write("2. XGBoost - 3 configurations\n")
    f.write("3. LightGBM - 3 configurations\n")
    f.write(f"Total configurations tested: {len(results_df)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("TOP PERFORMERS BY CRITERION\n")
    f.write("="*80 + "\n\n")
    
    best_test_r2 = results_df.loc[results_df['Test_R2'].idxmax()]
    f.write(f"1. Highest Test R²: {best_test_r2['Configuration']}\n")
    f.write(f"   Model Type: {best_test_r2['Model_Type']}\n")
    f.write(f"   Test R² = {best_test_r2['Test_R2']:.4f}\n")
    f.write(f"   CV R² = {best_test_r2['CV_R2_Mean']:.4f} ± {best_test_r2['CV_R2_Std']:.4f}\n\n")
    
    best_cv_r2 = results_df.loc[results_df['CV_R2_Mean'].idxmax()]
    f.write(f"2. Highest CV R²: {best_cv_r2['Configuration']}\n")
    f.write(f"   Model Type: {best_cv_r2['Model_Type']}\n")
    f.write(f"   CV R² = {best_cv_r2['CV_R2_Mean']:.4f} ± {best_cv_r2['CV_R2_Std']:.4f}\n")
    f.write(f"   Test R² = {best_cv_r2['Test_R2']:.4f}\n\n")
    
    best_pearson = results_df.loc[results_df['Test_Pearson_R'].idxmax()]
    f.write(f"3. Highest Test Pearson: {best_pearson['Configuration']}\n")
    f.write(f"   Model Type: {best_pearson['Model_Type']}\n")
    f.write(f"   Test Pearson = {best_pearson['Test_Pearson_R']:.4f}\n\n")
    
    best_generalization = results_df.loc[results_df['R2_Gap'].idxmin()]
    f.write(f"4. Best Generalization: {best_generalization['Configuration']}\n")
    f.write(f"   Model Type: {best_generalization['Model_Type']}\n")
    f.write(f"   R² Gap = {best_generalization['R2_Gap']:.4f}\n\n")
    
    best_consistency = results_df.loc[results_df['R2_Std_CellLines'].idxmin()]
    f.write(f"5. Most Consistent: {best_consistency['Configuration']}\n")
    f.write(f"   Model Type: {best_consistency['Model_Type']}\n")
    f.write(f"   R² Std = {best_consistency['R2_Std_CellLines']:.4f}\n\n")
    
    best_composite = results_df.loc[results_df['Composite_Score_2'].idxmax()]
    f.write(f"6. Best Overall: {best_composite['Configuration']}\n")
    f.write(f"   Model Type: {best_composite['Model_Type']}\n")
    f.write(f"   Composite = {best_composite['Composite_Score_2']:.4f}\n")
    f.write(f"   Test R² = {best_composite['Test_R2']:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("OVERALL PERFORMANCE SUMMARY (Sorted by Test R²)\n")
    f.write("="*80 + "\n\n")
    summary_cols = ['Configuration', 'Model_Type', 'CV_R2_Mean', 'CV_R2_Std', 
                    'Test_R2', 'R2_Gap', 'Test_Pearson_R', 'Composite_Score_2']
    f.write(results_df[summary_cols].sort_values('Test_R2', ascending=False).to_string(index=False))
    
    f.write("\n\n" + "="*80 + "\n")
    f.write("MODEL TYPE COMPARISON\n")
    f.write("="*80 + "\n\n")
    f.write(model_type_comparison.to_string())
    
    f.write("\n\n" + "="*80 + "\n")
    f.write("CV RELIABILITY ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Mean CV-Test R² difference: {results_df['CV_Test_R2_Diff'].mean():.4f}\n")
    f.write(f"Max CV-Test R² difference: {results_df['CV_Test_R2_Diff'].max():.4f}\n")
    f.write(f"Min CV-Test R² difference: {results_df['CV_Test_R2_Diff'].min():.4f}\n")

print(f"✓ Saved: {report_path}")

# ===== 4. Save Best Model =====
print("\n[4/6] Training and saving best model...")

# Get best configuration based on composite score
best_config_name = best_composite['Configuration']
best_config = boosting_configs[best_config_name]
best_model_type = best_config['model_type']
best_params = best_config['params']

print(f"Best configuration: {best_config_name}")
print(f"  Model Type: {best_model_type}")
print(f"  Test R² = {best_composite['Test_R2']:.4f}")
print(f"  CV R² = {best_composite['CV_R2_Mean']:.4f} ± {best_composite['CV_R2_Std']:.4f}")
print(f"  Composite Score = {best_composite['Composite_Score_2']:.4f}")

# Train best model on full training set
if best_model_type == 'GradientBoosting':
    best_model = GradientBoostingRegressor(**best_params)
elif best_model_type == 'XGBoost':
    best_model = XGBRegressor(**best_params)
elif best_model_type == 'LightGBM':
    best_model = LGBMRegressor(**best_params)

print("Training best model on full training set...")
best_model.fit(X_train, y_train)

# Save model
model_path = os.path.join(MODELS_DIR, "best_boosting_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✓ Saved: {model_path}")

# Save model metadata
metadata = {
    'configuration_name': best_config_name,
    'model_type': best_model_type,
    'parameters': best_params,
    'cv_r2_mean': best_composite['CV_R2_Mean'],
    'cv_r2_std': best_composite['CV_R2_Std'],
    'train_r2': best_composite['Train_R2'],
    'test_r2': best_composite['Test_R2'],
    'test_pearson': best_composite['Test_Pearson_R'],
    'r2_gap': best_composite['R2_Gap'],
    'composite_score': best_composite['Composite_Score_2'],
    'feature_names': feature_cols,
    'n_features': len(feature_cols),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'cv_strategy': '5-Fold GroupKFold',
    'train_date': str(pd.Timestamp.now())
}

metadata_path = os.path.join(MODELS_DIR, "best_model_metadata.pkl")
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Saved: {metadata_path}")

# ===== 5. Save All Trained Models (Optional - Top 3) =====
print("\n[5/6] Saving top 3 models...")

# Get top 3 configurations by composite score
top3_configs = results_df.nlargest(3, 'Composite_Score_2')

for idx, row in top3_configs.iterrows():
    config_name = row['Configuration']
    config = boosting_configs[config_name]
    model_type = config['model_type']
    params = config['params']
    
    # Train model
    if model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(**params)
    elif model_type == 'XGBoost':
        model = XGBRegressor(**params)
    elif model_type == 'LightGBM':
        model = LGBMRegressor(**params)
    
    model.fit(X_train, y_train)
    
    # Save model
    rank = list(top3_configs.index).index(idx) + 1
    model_filename = f"top{rank}_{config_name}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Saved Top {rank}: {model_filename} (Test R²={row['Test_R2']:.4f})")

# ===== 6. Save Configuration Parameters =====
print("\n[6/6] Saving all configuration parameters...")

config_path = os.path.join(RESULTS_DIR, "all_configurations.pkl")
with open(config_path, 'wb') as f:
    pickle.dump(boosting_configs, f)

print(f"✓ Saved: {config_path}")

# Save train/test split info (reference to existing split)
split_info = {
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_genes': len(set(groups_train)),
    'test_genes': len(set(groups_test)),
    'split_method': 'GroupShuffleSplit',
    'test_ratio': 0.2,
    'random_state': 42,
    'cv_method': '5-Fold GroupKFold',
    'note': 'Train/test indices saved in bagging analysis'
}

split_info_path = os.path.join(RESULTS_DIR, "split_info.pkl")
with open(split_info_path, 'wb') as f:
    pickle.dump(split_info, f)

print(f"✓ Saved: {split_info_path}")

# ===== Summary =====
print("\n" + "="*80)
print("✓ ALL DATA SAVED SUCCESSFULLY!")
print("="*80)

print(f"\nResults saved to: {RESULTS_DIR}/")
print("  • boosting_configurations_results.csv - All configuration metrics with CV")
print("  • boosting_per_cell_line_results.csv - Cell line specific performance")
print("  • model_type_comparison.csv - Comparison of GB, XGB, LGBM")
print("  • boosting_summary_report.txt - Human-readable comprehensive report")
print("  • all_configurations.pkl - All tested parameter configurations")
print("  • split_info.pkl - Data split information")

print(f"\nModels saved to: {MODELS_DIR}/")
print("  • best_boosting_model.pkl - Best performing model (trained)")
print("  • best_model_metadata.pkl - Best model metadata and parameters")
print("  • top1_*.pkl - Rank 1 model")
print("  • top2_*.pkl - Rank 2 model")
print("  • top3_*.pkl - Rank 3 model")

print(f"\nFigures saved to: {FIGURES_DIR}/")
print("  • 01_performance_dashboard.png")
print("  • 02_model_type_comparison.png")
print("  • 03_cell_line_heatmaps.png")
print("  • 04_cell_line_analysis.png")
print("  • 05_multi_criteria_ranking.png")

print("\n" + "="*80)

# ===== Final Summary Statistics =====
print("\n" + "="*80)
print("FINAL SUMMARY STATISTICS")
print("="*80)

print(f"\n📊 Best Performing Configuration:")
print(f"   Name: {best_config_name}")
print(f"   Type: {best_model_type}")
print(f"   Test R²: {best_composite['Test_R2']:.4f}")
print(f"   CV R²: {best_composite['CV_R2_Mean']:.4f} ± {best_composite['CV_R2_Std']:.4f}")
print(f"   Test Pearson: {best_composite['Test_Pearson_R']:.4f}")
print(f"   R² Gap: {best_composite['R2_Gap']:.4f}")
print(f"   Composite Score: {best_composite['Composite_Score_2']:.4f}")

print(f"\n📈 Model Type Winner:")
best_model_type_overall = model_type_comparison['Test_R2']['mean'].idxmax()
print(f"   {best_model_type_overall} has best average Test R²: "
      f"{model_type_comparison.loc[best_model_type_overall, ('Test_R2', 'mean')]:.4f}")

print(f"\n🎯 CV Reliability:")
print(f"   Average CV-Test R² difference: {results_df['CV_Test_R2_Diff'].mean():.4f}")
print(f"   All differences < 0.05: {(results_df['CV_Test_R2_Diff'] < 0.05).sum()}/{len(results_df)} configs")

print("\n" + "="*80)
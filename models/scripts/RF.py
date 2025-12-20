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
from sklearn.ensemble import RandomForestRegressor
import joblib


# Load data
merged = pd.read_csv("models/data/combined_data.csv",
                      header=0, sep=",")

bin_cols = [c for c in merged.columns if "_bin" in c]
X = merged[bin_cols]
y = merged["expression"]
groups = merged["gene_id"]

# Single split: Train vs Test (80/20 split)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

# Create train set
X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
groups_train = groups.iloc[train_idx]

# Create test set
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
groups_test = groups.iloc[test_idx]

# Verify no gene overlap
print(f"\n Split Results:")
print(f"  Train: {len(X_train)} samples, {groups_train.nunique()} genes")
print(f"  Test:  {len(X_test)} samples, {groups_test.nunique()} genes")

# Verify no gene overlap between train and test
overlap = set(groups_train.unique()) & set(groups_test.unique())
print(f"  Gene overlap: {len(overlap)} (should be 0)")

# Convert to numpy arrays for sklearn
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

print(f"\n Ready for modeling:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Train Random Forest Model
# ═══════════════════════════════════════════════════════════════════════

def tune_rf_with_cv(X_train, y_train, groups_train, 
                   cv_folds=5, n_jobs=-1, 
                   save_path="analysis/figures/"):
    """
    Hyperparameter tuning using CV on training set only
    Test set remains completely untouched until final evaluation
    
    Parameters:
    - X_train: Training features (numpy array)
    - y_train: Training labels (numpy array)
    - groups_train: Gene IDs for GroupKFold (pandas Series or numpy array)
    - cv_folds: Number of cross-validation folds (default=5)
    - n_jobs: Number of parallel jobs (-1 = use all cores)
    - save_path: Where to save results
    
    Returns:
    - best_model: Trained model with best hyperparameters
    - best_params: Dictionary of best hyperparameters
    - cv_results: Full CV results DataFrame
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*70)
    print("RANDOM FOREST HYPERPARAMETER TUNING (CV ON TRAINING SET)")
    print("="*70)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    total_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['min_samples_split']) * 
                         len(param_grid['min_samples_leaf']) * 
                         len(param_grid['max_features']))
    
    total_fits = total_combinations * cv_folds
    
    print(f"\nConfiguration:")
    print(f"  Parameter combinations: {total_combinations}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Total model fits: {total_fits}")
    print(f"  Training samples: {len(X_train)}")
    
    print(f"\n  Parameter grid:")
    for param, values in param_grid.items():
        print(f"    {param}: {values}")
    
    print(f"\nEstimated time: ~{total_fits * 1 / 60:.0f}-{total_fits * 3 / 60:.0f} minutes")
    print(f"     (depends on data size and hardware)")
    
    # Initialize base model
    rf_base = RandomForestRegressor(random_state=42, n_jobs=1)
    
    # Initialize GroupKFold (prevents gene leakage across CV folds)
    group_kfold = GroupKFold(n_splits=cv_folds)
    
    # Initialize GridSearchCV
    print(f"\nStarting Grid Search with {cv_folds}-Fold CV...")
    print(f"   Using GroupKFold to prevent gene leakage across folds")
    
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=group_kfold,
        scoring='r2',
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train, groups=groups_train)
    
    print("\nGrid Search Complete!")
    
    # Extract CV results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results_sorted = cv_results.sort_values('mean_test_score', ascending=False)
    
    # Save full CV results
    cv_results_sorted.to_csv(f"{save_path}rf_cv_results.csv", index=False)
    print(f"\nSaved full CV results to: {save_path}rf_cv_results.csv")
    
    # Display top 10 parameter combinations
    print("\n" + "="*70)
    print("TOP 10 PARAMETER COMBINATIONS (by CV R²)")
    print("="*70)
    
    display_cols = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split',
                   'param_min_samples_leaf', 'param_max_features',
                   'mean_train_score', 'mean_test_score', 'std_test_score']
    
    print(cv_results_sorted[display_cols].head(10).to_string(index=False))
    
    # Best parameters
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_cv_std = cv_results_sorted.iloc[0]['std_test_score']
    
    print("\n" + "="*70)
    print("BEST PARAMETERS (selected by CV)")
    print("="*70)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n  Cross-Validation Performance:")
    print(f"    Mean CV R²: {best_cv_score:.4f}")
    print(f"    Std CV R²: {best_cv_std:.4f}")
    print(f"    CV R² Range: [{best_cv_score - best_cv_std:.4f}, {best_cv_score + best_cv_std:.4f}]")
    
    # Get best model (already retrained on full training data)
    best_model = grid_search.best_estimator_
    
    # Check training performance
    y_train_pred = best_model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    print(f"\n  Full Training Set Performance:")
    print(f"    Training R²: {train_r2:.4f}")
    print(f"    Training RMSE: {train_rmse:.4f}")
    
    # Overfitting check
    overfitting_gap = train_r2 - best_cv_score
    print(f"\n  Overfitting Analysis:")
    print(f"    Gap (Train R² - CV R²): {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.05:
        print(f"    Minimal overfitting - model generalizes well")
    elif overfitting_gap < 0.10:
        print(f"    Moderate overfitting - acceptable but watch for this")
    else:
        print(f"    Significant overfitting - consider regularization")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Hyperparameters selected using {cv_folds}-fold CV")
    print(f"Best model trained on full training set")
    print(f"Test set remains untouched - use only for FINAL evaluation")
    print(f"\nAll results saved to {save_path}")
    
    return best_model, best_params, cv_results_sorted

best_rf_model, best_params, cv_results = tune_rf_with_cv(
    X_train=X_train,
    y_train=y_train,
    groups_train=groups_train,  # This is the pandas Series, NOT the numpy array
    cv_folds=5,
    n_jobs=24,
    save_path="models/results/rf/"
)

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Assess Random Forest Model
# ═══════════════════════════════════════════════════════════════════════

def assess_rf_model(model, X_train, y_train, X_test, y_test, 
                   groups_train, groups_test, feature_names,
                   save_path="analysis/figures/"):
    """
    Comprehensive assessment of Random Forest model
    
    Evaluates model on both training and test sets with:
    - Performance metrics (R², RMSE, MAE, correlations)
    - Prediction scatter plots
    - Residual analysis
    - Error distribution
    - Per-gene performance analysis
    
    Parameters:
    - model: Trained RandomForestRegressor
    - X_train: Training features (numpy array)
    - y_train: Training labels (numpy array)
    - X_test: Test features (numpy array)
    - y_test: Test labels (numpy array)
    - groups_train: Training gene IDs (pandas Series or array)
    - groups_test: Test gene IDs (pandas Series or array)
    - feature_names: List of feature names
    - save_path: Where to save results
    
    Returns:
    - Dictionary with all metrics and predictions
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL ASSESSMENT")
    print("="*70)
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. MAKE PREDICTIONS
    # ═══════════════════════════════════════════════════════════════════
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("  Predictions complete!")
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. CALCULATE METRICS
    # ═══════════════════════════════════════════════════════════════════
    print("\nCalculating performance metrics...")
    
    def calculate_metrics(y_true, y_pred, set_name):
        """Calculate comprehensive metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Correlations
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        # Residuals
        residuals = y_true - y_pred
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'mape': mape
        }
        
        print(f"\n  {set_name} Set Metrics:")
        print(f"    R² Score: {r2:.4f}")
        print(f"    Pearson Correlation: {pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"    Spearman Correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    MSE: {mse:.4f}")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    Residual Mean: {residual_mean:.4f}")
        print(f"    Residual Std: {residual_std:.4f}")
        
        return metrics
    
    train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
    test_metrics = calculate_metrics(y_test, y_test_pred, "Test")
    
    # Overfitting check
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS")
    print("="*70)
    overfitting_gap = train_metrics['r2'] - test_metrics['r2']
    print(f"  Train R²: {train_metrics['r2']:.4f}")
    print(f"  Test R²: {test_metrics['r2']:.4f}")
    print(f"  Gap: {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.05:
        print(f"  Minimal overfitting - excellent generalization!")
    elif overfitting_gap < 0.10:
        print(f"  Moderate overfitting - acceptable")
    else:
        print(f"  Significant overfitting - model may not generalize well")
    
    # Create metrics comparison table
    metrics_df = pd.DataFrame({
        'Train': train_metrics,
        'Test': test_metrics,
        'Difference': {k: train_metrics[k] - test_metrics[k] for k in train_metrics.keys()}
    }).T
    
    metrics_df.to_csv(f"{save_path}rf_metrics_comparison.csv")
    print(f"\nSaved metrics to: {save_path}rf_metrics_comparison.csv")
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. VISUALIZATION: Prediction Scatter Plots
    # ═══════════════════════════════════════════════════════════════════
    print("\nCreating prediction scatter plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = [
        ('Training', y_train, y_train_pred, train_metrics, axes[0]),
        ('Test', y_test, y_test_pred, test_metrics, axes[1])
    ]
    
    for name, y_true, y_pred, metrics, ax in datasets:
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.3)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.8)
        
        # Fitted line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), "b-", linewidth=2.5, alpha=0.8, 
               label=f'Fitted Line (y={z[0]:.3f}x+{z[1]:.3f})')
        
        # Labels
        ax.set_xlabel('True Expression', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted Expression', fontsize=13, fontweight='bold')
        ax.set_title(f'{name} Set (n={len(y_true)})', fontsize=15, fontweight='bold')
        
        # Metrics text box
        textstr = f'R² = {metrics["r2"]:.4f}\n'
        textstr += f'Pearson r = {metrics["pearson_r"]:.4f}\n'
        textstr += f'RMSE = {metrics["rmse"]:.4f}\n'
        textstr += f'MAE = {metrics["mae"]:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Random Forest: Predicted vs True Expression', 
                fontsize=17, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_path}rf_prediction_scatter.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. VISUALIZATION: Residual Analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\nCreating residual plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Residuals vs Predicted (Training)
    train_residuals = y_train - y_train_pred
    axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.5, s=30, 
                      edgecolors='k', linewidth=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Expression', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Residuals', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Set - Residuals vs Predicted', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Predicted (Test)
    test_residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.5, s=30, 
                      edgecolors='k', linewidth=0.3, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Expression', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Test Set - Residuals vs Predicted', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residual Distribution (Training)
    axes[1, 0].hist(train_residuals, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Training Set - Residual Distribution\nMean={train_residuals.mean():.4f}, Std={train_residuals.std():.4f}', 
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Residual Distribution (Test)
    axes[1, 1].hist(test_residuals, bins=50, alpha=0.7, edgecolor='black', color='coral')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'Test Set - Residual Distribution\nMean={test_residuals.mean():.4f}, Std={test_residuals.std():.4f}', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Random Forest: Residual Analysis', fontsize=17, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_path}rf_residual_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. SAVE PREDICTIONS
    # ═══════════════════════════════════════════════════════════════════
    print("\nSaving predictions...")
    
    # Training predictions
    train_results = pd.DataFrame({
        'gene_id': groups_train,
        'true_expression': y_train,
        'predicted_expression': y_train_pred,
        'residual': train_residuals,
        'absolute_error': np.abs(train_residuals),
        'squared_error': train_residuals ** 2
    })
    train_results.to_csv(f"{save_path}rf_train_predictions.csv", index=False)
    print(f"  ✓ Saved training predictions: {save_path}rf_train_predictions.csv")
    
    # Test predictions
    test_results = pd.DataFrame({
        'gene_id': groups_test,
        'true_expression': y_test,
        'predicted_expression': y_test_pred,
        'residual': test_residuals,
        'absolute_error': np.abs(test_residuals),
        'squared_error': test_residuals ** 2
    })
    test_results.to_csv(f"{save_path}rf_test_predictions.csv", index=False)
    print(f"  ✓ Saved test predictions: {save_path}rf_test_predictions.csv")
    
    # ═══════════════════════════════════════════════════════════════════
    # 6. FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("ASSESSMENT SUMMARY")
    print("="*70)
    print(f"\nModel evaluated on {len(y_train)} training and {len(y_test)} test samples")
    print(f"Test R² = {test_metrics['r2']:.4f}")
    print(f"Test Pearson r = {test_metrics['pearson_r']:.4f}")
    print(f"All visualizations and predictions saved to {save_path}")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'metrics_df': metrics_df,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_results': train_results,
        'test_results': test_results
    }

assessment_results = assess_rf_model(
    model=best_rf_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    groups_train=groups_train,
    groups_test=groups_test,
    feature_names=bin_cols,
    save_path="analysis/figures/"
)
joblib.dump(best_rf_model, 'models/trained/rf_model.pkl')

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Interpret Random Forest Model
# ═══════════════════════════════════════════════════════════════════════
def analyze_rf_feature_importance(model, feature_names, upstream_end, downstream_start,
                                  save_path="analysis/figures/"):
    """
    Comprehensive feature importance analysis for Random Forest
    
    Analyzes which features (histone marks and genomic regions) are most important
    for predicting gene expression
    
    Parameters:
    - model: Trained RandomForestRegressor
    - feature_names: List of feature names (bin columns)
    - upstream_end: Bin number where upstream region ends (5% boundary)
    - downstream_start: Bin number where downstream region starts (105% boundary)
    - save_path: Where to save results
    
    Returns:
    - Dictionary with importance DataFrames and analysis results
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*70)
    print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. EXTRACT FEATURE IMPORTANCES
    # ═══════════════════════════════════════════════════════════════════
    print("\nExtracting feature importances...")
    
    importances = model.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Extract histone mark (everything before "_bin")
    importance_df['Histone_Mark'] = importance_df['Feature'].str.split('_bin').str[0]
    
    # Extract bin number
    importance_df['Bin'] = importance_df['Feature'].str.extract(r'_bin(\d+)')[0].astype(int)
    
    print(f"  ✓ Extracted importances for {len(importance_df)} features")
    print(f"  ✓ Histone marks: {importance_df['Histone_Mark'].unique().tolist()}")
    print(f"  ✓ Bin range: {importance_df['Bin'].min()} to {importance_df['Bin'].max()}")
    
    # Pivot to create matrix: Histone_Mark × Bin
    importance_matrix = importance_df.pivot(index='Histone_Mark', columns='Bin', values='Importance')
    
    print(f"  ✓ Importance matrix shape: {importance_matrix.shape}")
    
    # Save importance values
    importance_df.to_csv(f"{save_path}rf_feature_importances.csv", index=False)
    importance_matrix.to_csv(f"{save_path}rf_importance_matrix.csv")
    print(f"\nSaved importances to CSV files")
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. OVERALL HISTONE MARK IMPORTANCE
    # ═══════════════════════════════════════════════════════════════════
    print("\nAnalyzing overall histone mark importance...")
    
    # Sum importance across all bins for each histone mark
    mark_importance = importance_df.groupby('Histone_Mark')['Importance'].sum().reset_index()
    mark_importance.columns = ['Histone_Mark', 'Total_Importance']
    mark_importance = mark_importance.sort_values('Total_Importance', ascending=False)
    
    # Also calculate mean importance
    mark_importance_mean = importance_df.groupby('Histone_Mark')['Importance'].mean().reset_index()
    mark_importance_mean.columns = ['Histone_Mark', 'Mean_Importance']
    
    mark_importance = pd.merge(mark_importance, mark_importance_mean, on='Histone_Mark')
    
    print("\n  Histone Mark Importance (Total and Mean):")
    print(mark_importance.to_string(index=False))
    
    mark_importance.to_csv(f"{save_path}rf_histone_mark_importance.csv", index=False)
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. REGIONAL IMPORTANCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\nAnalyzing regional importance...")
    
    total_bins = importance_matrix.shape[1]
    
    # Define regions based on 5% upstream/downstream
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
    
    # Calculate importance by region for each histone mark
    regional_importance = {}
    
    for mark in importance_matrix.index:
        regional_importance[mark] = {}
        for region_name, (start, end) in regions.items():
            region_imp = importance_matrix.loc[mark, start:end].sum()
            regional_importance[mark][region_name] = region_imp
    
    regional_importance_df = pd.DataFrame(regional_importance).T
    
    print("\n  Regional Importance (Total per region):")
    print(regional_importance_df.round(4).to_string())
    
    regional_importance_df.to_csv(f"{save_path}rf_regional_importance.csv")
    
    return importance_df, importance_matrix, mark_importance, regional_importance_df


def plot_rf_importance_heatmap(importance_matrix, upstream_end, downstream_start,
                               save_path="analysis/figures/"):
    """
    Create heatmap of feature importances across bins and histone marks
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nCreating importance heatmap...")
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Create heatmap
    sns.heatmap(importance_matrix, 
                cmap='YlOrRd',  # Yellow-Orange-Red colormap for importance
                cbar_kws={'label': 'Feature Importance'},
                xticklabels=50,  # Show every 50th bin
                yticklabels=True,
                ax=ax)
    
    # Add region boundary markers
    ax.axvline(x=upstream_end, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(x=downstream_start, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Bin Position (5% Upstream ← Gene Body (100%) → 5% Downstream)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Histone Mark', fontsize=14, fontweight='bold')
    ax.set_title('Random Forest: Feature Importance Across Gene Region\n5% Gene Length Upstream + Gene Body + 5% Gene Length Downstream', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add text annotations for regions
    ax.text(upstream_end/2, -0.5, 'Upstream\n(5% gene length)', 
           ha='center', fontsize=10, fontweight='bold', color='darkblue')
    ax.text((upstream_end + downstream_start)/2, -0.5, 'Gene Body\n(100% length)', 
           ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    ax.text((downstream_start + importance_matrix.shape[1])/2, -0.5, 'Downstream\n(5% gene length)', 
           ha='center', fontsize=10, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    filename = f"{save_path}rf_importance_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()


def plot_rf_importance_profiles(importance_matrix, upstream_end, downstream_start,
                                save_path="analysis/figures/"):
    """
    Plot importance profiles across gene region for each histone mark
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nCreating importance profiles...")
    
    fig, axes = plt.subplots(len(importance_matrix), 1, 
                            figsize=(16, 3*len(importance_matrix)), sharex=True)
    
    # If only one histone mark, make axes a list
    if len(importance_matrix) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(range(len(importance_matrix)))
    
    total_bins = importance_matrix.shape[1]
    
    for idx, (mark, color) in enumerate(zip(importance_matrix.index, colors)):
        ax = axes[idx]
        
        # Plot importance profile
        bins = importance_matrix.columns
        values = importance_matrix.loc[mark].values
        
        ax.fill_between(bins, 0, values, color=color, alpha=0.4)
        ax.plot(bins, values, color=color, linewidth=2.5, alpha=0.9)
        
        # Mark regions
        ax.axvspan(0, upstream_end, alpha=0.1, color='blue', label='Upstream (5%)')
        ax.axvspan(upstream_end, downstream_start, alpha=0.1, color='green')
        ax.axvspan(downstream_start, total_bins, alpha=0.1, color='orange', 
                  label='Downstream (5%)')
        
        # Add region boundaries
        ax.axvline(x=upstream_end, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(x=downstream_start, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        
        # Labels
        ax.set_ylabel('Importance', fontsize=11, fontweight='bold')
        ax.set_title(f'{mark}', fontsize=13, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        total_imp = values.sum()
        max_imp = values.max()
        max_bin = bins[np.argmax(values)]
        textstr = f'Total: {total_imp:.4f}\nMax: {max_imp:.4f} (bin {max_bin})'
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, ncol=2)
    
    axes[-1].set_xlabel('Bin Position (5% Upstream ← Gene Body (100%) → 5% Downstream)', 
                       fontsize=12, fontweight='bold')
    
    plt.suptitle('Random Forest: Feature Importance Profiles Across Gene Region\n5% Gene Length Upstream + Gene Body + 5% Gene Length Downstream', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f"{save_path}rf_importance_profiles.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()


def plot_rf_histone_mark_importance(mark_importance, save_path="analysis/figures/"):
    """
    Compare overall importance of different histone marks
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nPlotting histone mark importance comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.Set3(range(len(mark_importance)))
    
    # Plot 1: Total importance (sum across bins)
    bars1 = axes[0].bar(range(len(mark_importance)), 
                       mark_importance['Total_Importance'],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[0].set_xticks(range(len(mark_importance)))
    axes[0].set_xticklabels(mark_importance['Histone_Mark'], rotation=45, ha='right')
    axes[0].set_ylabel('Total Importance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    axes[0].set_title('Overall Feature Importance by Histone Mark\n(Sum across all bins)', 
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars1, mark_importance['Total_Importance'])):
        height = bar.get_height()
        axes[0].text(i, height, f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Mean importance (average per bin)
    bars2 = axes[1].bar(range(len(mark_importance)), 
                       mark_importance['Mean_Importance'],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[1].set_xticks(range(len(mark_importance)))
    axes[1].set_xticklabels(mark_importance['Histone_Mark'], rotation=45, ha='right')
    axes[1].set_ylabel('Mean Importance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    axes[1].set_title('Average Feature Importance by Histone Mark\n(Mean across all bins)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars2, mark_importance['Mean_Importance'])):
        height = bar.get_height()
        axes[1].text(i, height, f'{val:.6f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Random Forest: Histone Mark Importance Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{save_path}rf_histone_importance.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()


def plot_rf_regional_importance(regional_importance_df, save_path="analysis/figures/"):
    """
    Analyze importance of different gene regions
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n📊 Plotting regional importance...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    regional_importance_df.plot(kind='bar', ax=ax, width=0.8, 
                               color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#ee5a6f', '#c8d6e5'])
    
    ax.set_xlabel('Histone Mark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Importance', fontsize=12, fontweight='bold')
    ax.set_title('Regional Importance Analysis\n(Which regions contribute most to prediction?)', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Gene Region', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    filename = f"{save_path}rf_regional_importance.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()


def create_rf_importance_summary(importance_matrix, mark_importance, regional_importance_df,
                                 upstream_end, downstream_start, save_path="analysis/figures/"):
    """
    Create comprehensive summary figure for Random Forest feature importance
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nCreating comprehensive importance summary...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    total_bins = importance_matrix.shape[1]
    
    # Top: Heatmap (spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(importance_matrix, cmap='YlOrRd',
                xticklabels=50, yticklabels=True, ax=ax1,
                cbar_kws={'label': 'Importance'})
    ax1.axvline(x=upstream_end, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.axvline(x=downstream_start, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.set_title('A. Feature Importance Patterns Across Gene Region\n5% Upstream + Gene Body + 5% Downstream', 
                 fontsize=14, fontweight='bold', loc='left')
    ax1.set_xlabel('Bin Position (5% Upstream ← Gene Body → 5% Downstream)')
    ax1.set_ylabel('Histone Mark')
    
    # Second row: Regional importance
    ax2 = fig.add_subplot(gs[1, :])
    regional_importance_df.plot(kind='bar', ax=ax2, width=0.7, legend=True)
    ax2.set_title('B. Regional Importance Analysis', fontsize=13, fontweight='bold', loc='left')
    ax2.set_xlabel('Histone Mark', fontsize=11)
    ax2.set_ylabel('Total Importance', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(title='Gene Region', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    # Third row left: Total importance by histone mark
    ax3 = fig.add_subplot(gs[2, 0])
    colors = plt.cm.Set3(range(len(mark_importance)))
    bars = ax3.bar(range(len(mark_importance)), mark_importance['Total_Importance'],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(mark_importance)))
    ax3.set_xticklabels(mark_importance['Histone_Mark'], rotation=45, ha='right', fontsize=9)
    ax3.set_title('C. Total Importance by Mark', fontsize=13, fontweight='bold', loc='left')
    ax3.set_xlabel('Histone Mark', fontsize=11)
    ax3.set_ylabel('Total Importance', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, val in enumerate(mark_importance['Total_Importance']):
        ax3.text(i, val, f'{val:.3f}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
    
    # Third row right: Mean importance by histone mark
    ax4 = fig.add_subplot(gs[2, 1])
    bars = ax4.bar(range(len(mark_importance)), mark_importance['Mean_Importance'],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(mark_importance)))
    ax4.set_xticklabels(mark_importance['Histone_Mark'], rotation=45, ha='right', fontsize=9)
    ax4.set_title('D. Mean Importance by Mark', fontsize=13, fontweight='bold', loc='left')
    ax4.set_xlabel('Histone Mark', fontsize=11)
    ax4.set_ylabel('Mean Importance', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, val in enumerate(mark_importance['Mean_Importance']):
        ax4.text(i, val, f'{val:.5f}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold')
    
    # Bottom: Importance profiles (spans both columns)
    ax5 = fig.add_subplot(gs[3, :])
    plot_colors = plt.cm.Set2(range(len(importance_matrix)))
    for mark, color in zip(importance_matrix.index, plot_colors):
        ax5.plot(importance_matrix.columns, importance_matrix.loc[mark], 
                label=mark, color=color, linewidth=2.5, alpha=0.9)
        ax5.fill_between(importance_matrix.columns, 0, importance_matrix.loc[mark],
                        color=color, alpha=0.2)
    
    ax5.axvspan(0, upstream_end, alpha=0.1, color='blue')
    ax5.axvspan(upstream_end, downstream_start, alpha=0.1, color='green')
    ax5.axvspan(downstream_start, total_bins, alpha=0.1, color='orange')
    ax5.axvline(x=upstream_end, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax5.axvline(x=downstream_start, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    
    ax5.text(upstream_end/2, ax5.get_ylim()[1]*0.95, 'Upstream\n(5%)', 
            ha='center', fontsize=10, fontweight='bold', color='darkblue')
    ax5.text((upstream_end + downstream_start)/2, ax5.get_ylim()[1]*0.95, 'Gene Body\n(100%)', 
            ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    ax5.text((downstream_start + total_bins)/2, ax5.get_ylim()[1]*0.95, 'Downstream\n(5%)', 
            ha='center', fontsize=10, fontweight='bold', color='darkred')
    
    ax5.set_title('E. Importance Profiles Across Gene Region', fontsize=14, fontweight='bold', loc='left')
    ax5.set_xlabel('Bin Position (5% Upstream ← Gene Body (100%) → 5% Downstream)', fontsize=12)
    ax5.set_ylabel('Importance', fontsize=12)
    ax5.legend(loc='upper right', ncol=3, fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Random Forest: Feature Importance Summary\n5% Gene Length Upstream + Gene Body + 5% Gene Length Downstream', 
                fontsize=18, fontweight='bold')
    
    filename = f"{save_path}rf_importance_summary.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE - Complete Feature Importance Analysis
# ═══════════════════════════════════════════════════════════════════════

# First, you need to calculate region boundaries (same as for binning)
total_bins = len([c for c in bin_cols if c.startswith(bin_cols[0].split('_bin')[0])])
upstream_end = int(total_bins * 5/110)
downstream_start = int(total_bins * 105/110)

print(f"\nRegion boundaries:")
print(f"  Total bins per sample: {total_bins}")
print(f"  Upstream end: bin {upstream_end}")
print(f"  Downstream start: bin {downstream_start}")

# Run complete analysis
importance_df, importance_matrix, mark_importance, regional_importance_df = analyze_rf_feature_importance(
    model=best_rf_model,
    feature_names=bin_cols,
    upstream_end=upstream_end,
    downstream_start=downstream_start,
    save_path="models/results/rf"
)

# Create all visualizations
plot_rf_importance_heatmap(importance_matrix, upstream_end, downstream_start, 
                           save_path="analysis/figures/rf")

plot_rf_importance_profiles(importance_matrix, upstream_end, downstream_start,
                            save_path="analysis/figures/rf")

plot_rf_histone_mark_importance(mark_importance, save_path="analysis/figures/rf")

plot_rf_regional_importance(regional_importance_df, save_path="analysis/figures/rf")

create_rf_importance_summary(importance_matrix, mark_importance, regional_importance_df,
                             upstream_end, downstream_start, save_path="analysis/figures/rf")

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE!")
print("="*70)
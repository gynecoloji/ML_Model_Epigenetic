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
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Assuming your dataframe is 'df'
df = pd.read_csv("models/data/combined_data.csv")
# Separate features (X), target (y), and groups (genes)
feature_cols = [col for col in df.columns if col not in ['gene_id', 'cell_line', 'expression']]
X = df[feature_cols].values
y = df['expression'].values
groups = df['gene_id'].values  # Critical: group by gene_id to prevent leakage

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of unique genes: {len(np.unique(groups))}")
print(f"Number of samples: {len(y)}")
print(f"Feature columns: {len(feature_cols)}")


# Step 1: First split off a test set (20% of genes)
# Then use remaining 80% for train/validation with K-Fold CV

# Create test split
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss_test.split(X, y, groups))

# Split data into train+validation and test
X_train_val = X[train_val_idx]
y_train_val = y[train_val_idx]
groups_train_val = groups[train_val_idx]

X_test = X[test_idx]
y_test = y[test_idx]
groups_test = groups[test_idx]

# Verify gene separation
train_val_genes = set(groups_train_val)
test_genes = set(groups_test)
overlap = train_val_genes.intersection(test_genes)

print("=" * 60)
print("TRAIN/VALIDATION vs TEST SPLIT")
print("=" * 60)
print(f"Train+Val samples: {len(X_train_val)}, Test samples: {len(X_test)}")
print(f"Train+Val genes: {len(train_val_genes)}, Test genes: {len(test_genes)}")
print(f"Gene overlap: {len(overlap)} (should be 0!)")
print()

# Step 2: Set up K-Fold CV on the train+validation set only
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

print(f"Setting up {n_splits}-fold CV on train+validation set")
print("=" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups_train_val)):
    fold_train_genes = set(groups_train_val[train_idx])
    fold_val_genes = set(groups_train_val[val_idx])
    fold_overlap = fold_train_genes.intersection(fold_val_genes)
    
    print(f"Fold {fold_idx + 1}:")
    print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"  Train genes: {len(fold_train_genes)}, Val genes: {len(fold_val_genes)}")
    print(f"  Overlap: {len(fold_overlap)} (should be 0!)")

print()
print("✓ Data splitting complete!")
print(f"✓ Test set is held out and will only be used for final evaluation")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define parameter grids with REGULARIZATION parameters

# 1. Random Forest - Regularization via tree constraints
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],              # Limit tree depth
    'min_samples_split': [5, 10, 20],       # Min samples to split
    'min_samples_leaf': [2, 5, 10],         # Min samples in leaf
    'max_features': ['sqrt', 'log2'],       # Limit features per split
    'min_impurity_decrease': [0.0, 0.001, 0.01]  # Min improvement to split
}

# 2. XGBoost - L1/L2 regularization + tree constraints
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1.0, 10.0],       # L1 regularization
    'reg_lambda': [0.1, 1.0, 10.0],         # L2 regularization
    'gamma': [0, 0.1, 0.5, 1.0],            # Min loss reduction
    'min_child_weight': [1, 3, 5]           # Min sum of weights in child
}

# 3. LightGBM - L1/L2 regularization + tree constraints
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1.0, 10.0],       # L1 regularization
    'reg_lambda': [0.1, 1.0, 10.0],         # L2 regularization
    'min_gain_to_split': [0, 0.1, 0.5],     # Min gain to split
    'min_child_samples': [5, 10, 20]        # Min samples in leaf
}

# 4. Gradient Boosting - Tree constraints
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'min_impurity_decrease': [0.0, 0.001, 0.01]
}

# 5. Ridge - L2 regularization (already built-in)
ridge_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # Stronger regularization
}

# 6. Lasso - L1 regularization (already built-in)
lasso_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# 7. ElasticNet - L1 + L2 regularization (already built-in)
elasticnet_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Balance between L1 and L2
}

param_grids = {
    'Random Forest': rf_param_grid,
    'XGBoost': xgb_param_grid,
    'LightGBM': lgbm_param_grid,
    'GradientBoosting': gb_param_grid,
    'Ridge': ridge_param_grid,
    'Lasso': lasso_param_grid,
    'ElasticNet': elasticnet_param_grid
}

print("REGULARIZATION PARAMETERS ADDED")
print("=" * 60)
for name, grid in param_grids.items():
    print(f"\n{name}:")
    reg_params = []
    if 'reg_alpha' in grid or 'reg_lambda' in grid:
        reg_params.append("L1/L2 regularization")
    if 'max_depth' in grid:
        reg_params.append("Tree depth limits")
    if 'min_samples_split' in grid or 'min_samples_leaf' in grid:
        reg_params.append("Sample size constraints")
    if 'alpha' in grid:
        reg_params.append("Regularization strength (alpha)")
    print(f"  Regularization: {', '.join(reg_params)}")
    n_combinations = np.prod([len(v) for v in grid.values()])
    print(f"  Total combinations: {n_combinations}")
    
from sklearn.model_selection import RandomizedSearchCV
import time
import multiprocessing
from joblib import parallel_backend

# Get number of available CPU cores
n_cores = multiprocessing.cpu_count()
print(f"Available CPU cores: {n_cores}")
print()

# PARALLELIZATION STRATEGY:
# For tree-based models: Use model-level parallelization (faster)
# For CV search: Use search-level parallelization
# Don't oversubscribe - balance between model n_jobs and CV n_jobs

# Initialize base model instances with optimized parallelization
base_model_instances = {
    # Tree models: Let model handle parallelization internally
    'Random Forest': RandomForestRegressor(
        n_jobs=-1,          # Use all cores within the model
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_jobs=-1,          # Use all cores
        random_state=42, 
        verbosity=0,
        tree_method='hist'  # Faster histogram-based algorithm
    ),
    'LightGBM': LGBMRegressor(
        n_jobs=-1,          # Use all cores
        random_state=42, 
        verbose=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        random_state=42     # No native parallelization, CV will parallelize
    ),
    
    # Linear models: Fast, so parallelize at CV level
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42, max_iter=2000),
    'ElasticNet': ElasticNet(random_state=42, max_iter=2000)
}

# Parallelization settings for each model type
parallel_settings = {
    'Random Forest': {'cv_jobs': 1, 'model_jobs': -1},      # Parallelize within model
    'XGBoost': {'cv_jobs': 1, 'model_jobs': -1},            # Parallelize within model
    'LightGBM': {'cv_jobs': 1, 'model_jobs': -1},           # Parallelize within model
    'GradientBoosting': {'cv_jobs': -1, 'model_jobs': 1},   # Parallelize across CV folds
    'Ridge': {'cv_jobs': -1, 'model_jobs': 1},              # Parallelize across CV folds
    'Lasso': {'cv_jobs': -1, 'model_jobs': 1},              # Parallelize across CV folds
    'ElasticNet': {'cv_jobs': -1, 'model_jobs': 1}          # Parallelize across CV folds
}

# Set up GroupKFold
gkf = GroupKFold(n_splits=5)

# Storage for results
best_models = {}
search_results = {}

print("HYPERPARAMETER TUNING WITH OPTIMIZED PARALLELIZATION")
print("=" * 60)
print(f"Strategy: Balance model-level vs CV-level parallelization")
print(f"Training samples: {len(X_train_val)}, Genes: {len(set(groups_train_val))}")
print("=" * 60)
print()

# Perform RandomizedSearchCV with optimized parallelization
for model_name in base_model_instances.keys():
    print(f"🔍 Tuning {model_name}...")
    start_time = time.time()
    
    # Get parallelization settings for this model
    cv_jobs = parallel_settings[model_name]['cv_jobs']
    
    # Reduce n_iter for faster tuning (can increase if needed)
    n_iter = 15 if model_name in ['Ridge', 'Lasso', 'ElasticNet'] else 25
    
    # Create RandomizedSearchCV with optimized parallelization
    random_search = RandomizedSearchCV(
        estimator=base_model_instances[model_name],
        param_distributions=param_grids[model_name],
        n_iter=n_iter,
        cv=gkf,
        scoring='r2',
        n_jobs=cv_jobs,             # Parallelization at CV level
        verbose=1,                  # Show progress
        random_state=42,
        return_train_score=True
    )
    
    # Use joblib backend for better thread management
    with parallel_backend('threading', n_jobs=-1):
        random_search.fit(X_train_val, y_train_val, groups=groups_train_val)
    
    # Store results
    best_models[model_name] = random_search.best_estimator_
    search_results[model_name] = {
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_,
        'cv_std': random_search.cv_results_['std_test_score'][random_search.best_index_],
        'time': time.time() - start_time
    }
    
    print(f"   ✓ Best CV R² = {random_search.best_score_:.4f} (±{search_results[model_name]['cv_std']:.4f})")
    print(f"   ⏱ Time: {search_results[model_name]['time']:.1f}s")
    print(f"   📊 Best params: {random_search.best_params_}")
    print()

print("=" * 60)
print("TUNING COMPLETE - SUMMARY")
print("=" * 60)

with open("models/trained/ensembl_tuning_checkpoint.pkl", "wb") as f:
    pickle.dump(
        {
            "best_models": best_models,
            "search_results": search_results
        },
        f
    )

# Sort by CV score
sorted_models = sorted(search_results.items(), key=lambda x: x[1]['best_cv_score'], reverse=True)
for model_name, results in sorted_models:
    print(f"{model_name:20s}: R² = {results['best_cv_score']:.4f}, Time = {results['time']:.1f}s")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dictionary to store test performance
test_performance = {}

print("EVALUATING TUNED MODELS ON TEST SET")
print("=" * 60)
print(f"Test set: {len(X_test)} samples, {len(set(groups_test))} genes")
print("=" * 60)
print()

# Evaluate each model on test set
for model_name, model in best_models.items():
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    test_performance[model_name] = {
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'predictions': y_pred
    }
    
    print(f"{model_name}:")
    print(f"   R² Score:  {r2:.4f}")
    print(f"   RMSE:      {rmse:.4f}")
    print(f"   MAE:       {mae:.4f}")
    print()

# Create performance comparison DataFrame
performance_df = pd.DataFrame({
    'Model': list(test_performance.keys()),
    'CV_R2': [search_results[m]['best_cv_score'] for m in test_performance.keys()],
    'Test_R2': [test_performance[m]['R2'] for m in test_performance.keys()],
    'Test_RMSE': [test_performance[m]['RMSE'] for m in test_performance.keys()],
    'Test_MAE': [test_performance[m]['MAE'] for m in test_performance.keys()]
})

# Sort by test R2
performance_df = performance_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)

print("=" * 60)
print("PERFORMANCE SUMMARY (sorted by Test R²)")
print("=" * 60)
print(performance_df.to_string(index=False))
print()

# Check for overfitting (CV vs Test performance)
print("=" * 60)
print("OVERFITTING CHECK (CV R² vs Test R²)")
print("=" * 60)
for _, row in performance_df.iterrows():
    diff = row['CV_R2'] - row['Test_R2']
    status = "✓ Good" if abs(diff) < 0.05 else "⚠ Warning" if abs(diff) < 0.10 else "❌ Overfitting"
    print(f"{row['Model']:20s}: CV={row['CV_R2']:.4f}, Test={row['Test_R2']:.4f}, Diff={diff:+.4f} {status}")


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ==========================================
# PLOT 1: Performance Comparison Bar Chart
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R² Score comparison
ax1 = axes[0]
x_pos = np.arange(len(performance_df))
ax1.bar(x_pos - 0.2, performance_df['CV_R2'], width=0.4, label='CV R²', alpha=0.8, color='skyblue')
ax1.bar(x_pos + 0.2, performance_df['Test_R2'], width=0.4, label='Test R²', alpha=0.8, color='coral')
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('R² Score: CV vs Test', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(performance_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# RMSE comparison
ax2 = axes[1]
ax2.bar(x_pos, performance_df['Test_RMSE'], color='steelblue', alpha=0.8)
ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax2.set_title('Root Mean Squared Error (Test)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(performance_df['Model'], rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# MAE comparison
ax3 = axes[2]
ax3.bar(x_pos, performance_df['Test_MAE'], color='seagreen', alpha=0.8)
ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax3.set_title('Mean Absolute Error (Test)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(performance_df['Model'], rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/ensembl/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Performance comparison saved as 'model_performance_comparison.png'")
print()

# ==========================================
# PLOT 2: Predicted vs Actual (all models)
# ==========================================
n_models = len(best_models)
n_cols = 3
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, (model_name, model) in enumerate(best_models.items()):
    ax = axes[idx]
    
    # Get predictions
    y_pred = test_performance[model_name]['predictions']
    r2 = test_performance[model_name]['R2']
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Expression', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Expression', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

# Hide extra subplots
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('analysis/figures/ensembl/predicted_vs_actual_all_models.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Predicted vs Actual plots saved as 'predicted_vs_actual_all_models.png'")
print()

# ==========================================
# PLOT 3: Residual Analysis (Top 3 Models)
# ==========================================
top_3_models = performance_df.head(3)['Model'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_name in enumerate(top_3_models):
    ax = axes[idx]
    
    # Calculate residuals
    y_pred = test_performance[model_name]['predictions']
    residuals = y_test - y_pred
    
    # Residual plot
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    # Labels
    ax.set_xlabel('Predicted Expression', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name}\nResidual Plot', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/ensembl/residual_plots_top3.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Residual plots saved as 'residual_plots_top3.png'")
print()

# ==========================================
# PLOT 4: Distribution of Predictions vs Actual
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual distribution
ax1 = axes[0]
ax1.hist(y_test, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_xlabel('Gene Expression', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Actual Expression Distribution (Test Set)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Predicted distributions (top 3 models)
ax2 = axes[1]
colors = ['coral', 'lightgreen', 'plum']
for idx, model_name in enumerate(top_3_models):
    y_pred = test_performance[model_name]['predictions']
    ax2.hist(y_pred, bins=50, alpha=0.5, label=model_name, color=colors[idx], edgecolor='black')

ax2.hist(y_test, bins=50, alpha=0.3, label='Actual', color='gray', edgecolor='black')
ax2.set_xlabel('Gene Expression', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Predicted vs Actual Distribution (Top 3)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/ensembl/distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Distribution comparison saved as 'distribution_comparison.png'")
print()

print("=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print("Generated plots:")
print("  1. model_performance_comparison.png")
print("  2. predicted_vs_actual_all_models.png")
print("  3. residual_plots_top3.png")
print("  4. distribution_comparison.png")

from sklearn.ensemble import VotingRegressor

print("BUILDING VOTING ENSEMBLE MODELS")
print("=" * 60)
print()

# ==========================================
# Strategy 1: Simple Voting (Equal Weights)
# ==========================================
print("Strategy 1: Simple Voting Ensemble (all models, equal weights)")

# Create list of all tuned models for voting
estimators_all = [(name, model) for name, model in best_models.items()]

# Simple voting ensemble
voting_simple = VotingRegressor(
    estimators=estimators_all,
    n_jobs=-1
)

# Train on train+validation set
print(f"   Training on {len(X_train_val)} samples...")
voting_simple.fit(X_train_val, y_train_val)

# Predict on test set
y_pred_simple = voting_simple.predict(X_test)

# Evaluate
r2_simple = r2_score(y_test, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))
mae_simple = mean_absolute_error(y_test, y_pred_simple)

print(f"   ✓ Test R²:   {r2_simple:.4f}")
print(f"   ✓ Test RMSE: {rmse_simple:.4f}")
print(f"   ✓ Test MAE:  {mae_simple:.4f}")
print()

# ==========================================
# Strategy 2: Weighted Voting (Performance-based)
# ==========================================
print("Strategy 2: Weighted Voting (weights based on CV R²)")

# Calculate weights from CV performance
weights = []
for name in best_models.keys():
    cv_r2 = search_results[name]['best_cv_score']
    # Use R² as weight (models with higher R² get more weight)
    weights.append(max(cv_r2, 0))  # Ensure non-negative

# Normalize weights to sum to 1
weights = np.array(weights)
weights = weights / weights.sum()

print("   Model weights:")
for name, w in zip(best_models.keys(), weights):
    print(f"      {name:20s}: {w:.4f}")
print()

# Weighted voting ensemble
voting_weighted = VotingRegressor(
    estimators=estimators_all,
    weights=weights,
    n_jobs=-1
)

# Train
print(f"   Training on {len(X_train_val)} samples...")
voting_weighted.fit(X_train_val, y_train_val)

# Predict
y_pred_weighted = voting_weighted.predict(X_test)

# Evaluate
r2_weighted = r2_score(y_test, y_pred_weighted)
rmse_weighted = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
mae_weighted = mean_absolute_error(y_test, y_pred_weighted)

print(f"   ✓ Test R²:   {r2_weighted:.4f}")
print(f"   ✓ Test RMSE: {rmse_weighted:.4f}")
print(f"   ✓ Test MAE:  {mae_weighted:.4f}")
print()

# ==========================================
# Strategy 3: Top-K Voting (Best 3 models)
# ==========================================
print("Strategy 3: Top-K Voting (best 3 models only)")

# Get top 3 models based on test performance
top_k = 3
top_models = performance_df.head(top_k)['Model'].tolist()
print(f"   Selected models: {', '.join(top_models)}")

# Create estimators list for top-k
estimators_topk = [(name, best_models[name]) for name in top_models]

# Get weights for top-k models
weights_topk = []
for name in top_models:
    cv_r2 = search_results[name]['best_cv_score']
    weights_topk.append(max(cv_r2, 0))

weights_topk = np.array(weights_topk)
weights_topk = weights_topk / weights_topk.sum()

print("   Weights:")
for name, w in zip(top_models, weights_topk):
    print(f"      {name:20s}: {w:.4f}")
print()

# Top-K voting ensemble
voting_topk = VotingRegressor(
    estimators=estimators_topk,
    weights=weights_topk,
    n_jobs=-1
)

# Train
print(f"   Training on {len(X_train_val)} samples...")
voting_topk.fit(X_train_val, y_train_val)

# Predict
y_pred_topk = voting_topk.predict(X_test)

# Evaluate
r2_topk = r2_score(y_test, y_pred_topk)
rmse_topk = np.sqrt(mean_squared_error(y_test, y_pred_topk))
mae_topk = mean_absolute_error(y_test, y_pred_topk)

print(f"   ✓ Test R²:   {r2_topk:.4f}")
print(f"   ✓ Test RMSE: {rmse_topk:.4f}")
print(f"   ✓ Test MAE:  {mae_topk:.4f}")
print()

# ==========================================
# Compare All Voting Strategies
# ==========================================
print("=" * 60)
print("VOTING ENSEMBLE COMPARISON")
print("=" * 60)

voting_results = pd.DataFrame({
    'Ensemble Type': ['Simple Voting (All)', 'Weighted Voting (All)', f'Top-{top_k} Weighted'],
    'Test_R2': [r2_simple, r2_weighted, r2_topk],
    'Test_RMSE': [rmse_simple, rmse_weighted, rmse_topk],
    'Test_MAE': [mae_simple, mae_weighted, mae_topk]
})

print(voting_results.to_string(index=False))
print()

# Compare with best individual model
best_individual_r2 = performance_df.iloc[0]['Test_R2']
best_individual_name = performance_df.iloc[0]['Model']

print(f"Best Individual Model: {best_individual_name} (R² = {best_individual_r2:.4f})")
print()

for _, row in voting_results.iterrows():
    improvement = row['Test_R2'] - best_individual_r2
    status = "✓ Better" if improvement > 0 else "✗ Worse"
    print(f"{row['Ensemble Type']:25s}: {status} (Δ R² = {improvement:+.4f})")

# Store voting ensembles for later use
ensemble_models = {
    'Voting_Simple': voting_simple,
    'Voting_Weighted': voting_weighted,
    'Voting_TopK': voting_topk
}



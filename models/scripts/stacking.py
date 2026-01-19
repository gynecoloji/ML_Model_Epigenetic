import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import os

os.chdir("ML")
df = pd.read_csv("models/data/combined_data.csv")

# ============================================================================
# SECTION 1: DATA PREPARATION & GENE-BASED TRAIN/TEST SPLIT
# ============================================================================

X = df.drop(['gene_id', 'cell_line', 'expression'], axis=1)
y = df['expression']
genes = df['gene_id']

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=genes))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
genes_train = genes.iloc[train_idx]
genes_test = genes.iloc[test_idx]

# ============================================================================
# SECTION 2: BASE MODEL SETUP & GROUPKFOLD CROSS-VALIDATION
# ============================================================================

n_folds = 5
gkf = GroupKFold(n_splits = n_folds)

base_models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    
    'XGBoost': XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    ),
    
    'LightGBM': LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    
    'ElasticNet': ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42,
        max_iter=2000
    ),
    
    'Ridge': Ridge(
        alpha=1.0,
        random_state=42
    )
}

param_distributions = {
    'RandomForest': {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5]
    },
    
    'XGBoost': {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0]
    },
    
    'LightGBM': {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0]
    },
    
    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    
    'Ridge': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
}

tuned_base_models = {}
tuning_results = {}

for model_name, model in base_models.items():
    print(f"\n{'='*60}")
    print(f"Tuning: {model_name}")
    print(f"{'='*60}")
    
    # Setup RandomizedSearchCV with GroupKFold
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[model_name],
        n_iter=20,  # Try 20 random combinations
        cv=GroupKFold(n_splits=n_folds),  # Gene-based CV
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    # Fit with gene grouping
    print(f"  Searching 20 hyperparameter combinations...")
    random_search.fit(X_train, y_train, groups=genes_train)
    
    # Store results
    tuned_base_models[model_name] = random_search.best_estimator_
    tuning_results[model_name] = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_
    }
    
    print(f"  ✓ Best CV R²: {random_search.best_score_:.4f}")
    print(f"  Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"    {param}: {value}")

# ============================================================================
# SECTION 3: GENERATE OUT-OF-FOLD PREDICTIONS FOR META-MODEL
# ============================================================================

oof_predictions = np.zeros((len(X_train), len(base_models)))
test_predictions = np.zeros((len(X_test), len(base_models)))

print("\n" + "="*60)
print("GENERATING OUT-OF-FOLD PREDICTIONS (TUNED MODELS)")
print("="*60)

for model_idx, (model_name, model) in enumerate(tuned_base_models.items()):
    print(f"\n{'='*60}")
    print(f"Base Model {model_idx + 1}/{len(tuned_base_models)}: {model_name}")
    print(f"{'='*60}")
    
    model_test_preds = np.zeros((len(X_test), n_folds))
    
    # Loop through each fold
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(gkf.split(X_train, y_train, groups=genes_train)):
        print(f"  Fold {fold_idx + 1}/{n_folds}...", end=" ")
        
        # Split data for this fold
        X_fold_train = X_train.iloc[train_fold_idx]
        y_fold_train = y_train.iloc[train_fold_idx]
        X_fold_val = X_train.iloc[val_fold_idx]
        y_fold_val = y_train.iloc[val_fold_idx]
        
        # Train tuned model
        model.fit(X_fold_train, y_fold_train)
        
        # Out-of-fold predictions
        oof_pred = model.predict(X_fold_val)
        oof_predictions[val_fold_idx, model_idx] = oof_pred
        
        # Test predictions
        test_pred = model.predict(X_test)
        model_test_preds[:, fold_idx] = test_pred
        
        fold_r2 = r2_score(y_fold_val, oof_pred)
        print(f"R² = {fold_r2:.4f}")
    
    # Average test predictions across folds
    test_predictions[:, model_idx] = model_test_preds.mean(axis=1)
    
    # Overall OOF performance
    oof_r2 = r2_score(y_train, oof_predictions[:, model_idx])
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions[:, model_idx]))
    
    print(f"\n  Overall OOF Performance:")
    print(f"    R² = {oof_r2:.4f}")
    print(f"    RMSE = {oof_rmse:.4f}")

print("\n✓ Out-of-fold predictions ready for meta-model!")

from sklearn.model_selection import GridSearchCV

# ============================================================================
# SECTION 3.5: HYPERPARAMETER TUNING FOR META-MODEL
# ============================================================================

print("\n" + "="*60)
print("META-MODEL HYPERPARAMETER TUNING")
print("="*60)
print("Strategy: GridSearchCV on out-of-fold predictions")
print(f"CV Strategy: GroupKFold ({n_folds} folds, gene-based)")
print("="*60)

# Define meta-model parameter grid
meta_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
}

print(f"\nTesting {len(meta_param_grid['alpha'])} alpha values...")
print(f"Alpha range: {meta_param_grid['alpha'][0]} to {meta_param_grid['alpha'][-1]}")

# Create base Ridge model
base_meta_model = Ridge(random_state=42)

# Setup GridSearchCV with GroupKFold on OOF predictions
# CRITICAL: Use genes_train for grouping even though we're working with OOF predictions
meta_grid_search = GridSearchCV(
    estimator=base_meta_model,
    param_grid=meta_param_grid,
    cv=GroupKFold(n_splits=n_folds),  # Still use gene-based CV!
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Tune meta-model on out-of-fold predictions
print("\nTuning meta-model on out-of-fold predictions...")
meta_grid_search.fit(oof_predictions, y_train, groups=genes_train)

# Get best meta-model
best_meta_model = meta_grid_search.best_estimator_
best_alpha = meta_grid_search.best_params_['alpha']
best_cv_score = meta_grid_search.best_score_

print("\n" + "="*60)
print("META-MODEL TUNING RESULTS")
print("="*60)
print(f"Best alpha: {best_alpha}")
print(f"Best CV R²: {best_cv_score:.4f}")
print("="*60)

# Show performance across different alphas
print("\nPerformance across alpha values:")
results_df = pd.DataFrame(meta_grid_search.cv_results_)
for alpha, mean_score, std_score in zip(
    results_df['param_alpha'],
    results_df['mean_test_score'],
    results_df['std_test_score']
):
    print(f"  alpha = {alpha:8.3f}  →  R² = {mean_score:.4f} (±{std_score:.4f})")

print("\n✓ Meta-model tuned and ready!")

# ============================================================================
# SECTION 4: TRAIN TUNED META-MODEL ON OUT-OF-FOLD PREDICTIONS
# ============================================================================

print("\n" + "="*60)
print("META-MODEL TRAINING (Level 1) - TUNED")
print("="*60)

print(f"\nMeta-model input features: {len(tuned_base_models)} (one per base model)")
print(f"Meta-model training samples: {oof_predictions.shape[0]}")
print(f"Meta-model test samples: {test_predictions.shape[0]}")
print(f"Optimal alpha: {best_alpha}")  # Show the tuned parameter

# Use the TUNED meta-model (already fitted during GridSearchCV)
# But refit on FULL out-of-fold predictions for final model
meta_model = Ridge(alpha=best_alpha, random_state=42)

print("\nTraining tuned meta-model on full OOF predictions...")
meta_model.fit(oof_predictions, y_train)
print("✓ Meta-model training completed!")

# Make predictions on test set
y_pred_stacked = meta_model.predict(test_predictions)

# Evaluate stacked model performance
r2 = r2_score(y_test, y_pred_stacked)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_stacked))
mae = mean_absolute_error(y_test, y_pred_stacked)
pearson_r, _ = pearsonr(y_test, y_pred_stacked)
spearman_r, _ = spearmanr(y_test, y_pred_stacked)

print("\n" + "="*60)
print("STACKED ENSEMBLE PERFORMANCE ON TEST SET (TUNED)")
print("="*60)
print(f"R² Score:          {r2:.4f}")
print(f"RMSE:              {rmse:.4f}")
print(f"MAE:               {mae:.4f}")
print(f"Pearson r:         {pearson_r:.4f}")
print(f"Spearman r:        {spearman_r:.4f}")
print("="*60)

# Show meta-model coefficients
print("\n" + "="*60)
print("META-MODEL WEIGHTS (Base Model Importance)")
print("="*60)
model_names = list(tuned_base_models.keys())
coefficients = meta_model.coef_

for name, coef in zip(model_names, coefficients):
    print(f"{name:15s}: {coef:7.4f}")
print(f"\nIntercept:      {meta_model.intercept_:.4f}")
print(f"Alpha (tuned):  {best_alpha}")
print("="*60)

# Store results
stacked_results = {
    'predictions': y_pred_stacked,
    'r2': r2,
    'rmse': rmse,
    'mae': mae,
    'pearson': pearson_r,
    'spearman': spearman_r,
    'weights': dict(zip(model_names, coefficients)),
    'alpha': best_alpha
}

from scipy.stats import pearsonr, spearmanr

# ============================================================================
# SECTION 5: EVALUATE INDIVIDUAL MODELS ON TEST SET
# ============================================================================

print("\n" + "="*60)
print("EVALUATING INDIVIDUAL BASE MODELS ON TEST SET")
print("="*60)

individual_results = {}

for model_name, model in tuned_base_models.items():
    print(f"\nEvaluating: {model_name}...")
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    pearson_r, _ = pearsonr(y_test, y_pred)
    spearman_r, _ = spearmanr(y_test, y_pred)
    
    individual_results[model_name] = {
        'predictions': y_pred,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson_r,
        'spearman': spearman_r
    }
    
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}")

# Calculate improvement
best_individual_r2 = max([r['r2'] for r in individual_results.values()])
improvement = ((stacked_results['r2'] - best_individual_r2) / best_individual_r2) * 100

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"Best Individual R²:  {best_individual_r2:.4f}")
print(f"Stacked Ensemble R²: {stacked_results['r2']:.4f}")
print(f"Improvement:         {improvement:+.2f}%")
print("="*60)

# ============================================================================
# SECTION 6: SAVE MODELS AND RESULTS
# ============================================================================

import joblib
import json
from datetime import datetime

print("\n" + "="*60)
print("SAVING MODELS AND RESULTS")
print("="*60)

# Create save directory
save_dir = "models/saved_ensemble"
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"stacking_ensemble_{timestamp}"
experiment_dir = os.path.join(save_dir, experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

print(f"\nSaving to: {experiment_dir}/")

# ----------------------------------------------------------------------------
# 6.1: Save Tuned Base Models
# ----------------------------------------------------------------------------

print("\n1. Saving tuned base models...")
base_models_dir = os.path.join(experiment_dir, "base_models")
os.makedirs(base_models_dir, exist_ok=True)

for model_name, model in tuned_base_models.items():
    # Retrain on full training set before saving
    model.fit(X_train, y_train)
    model_path = os.path.join(base_models_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"   ✓ Saved: {model_name}.pkl")

# ----------------------------------------------------------------------------
# 6.2: Save Meta-Model
# ----------------------------------------------------------------------------

print("\n2. Saving meta-model...")
meta_model_path = os.path.join(experiment_dir, "meta_model.pkl")
joblib.dump(meta_model, meta_model_path)
print(f"   ✓ Saved: meta_model.pkl")

# ----------------------------------------------------------------------------
# 6.3: Save Hyperparameter Tuning Results
# ----------------------------------------------------------------------------

print("\n3. Saving hyperparameter tuning results...")

# Save full tuning results (includes cv_results)
tuning_results_path = os.path.join(experiment_dir, "tuning_results.pkl")
joblib.dump(tuning_results, tuning_results_path)
print(f"   ✓ Saved: tuning_results.pkl")

# Save best parameters as JSON
best_params = {model_name: results['best_params'] 
               for model_name, results in tuning_results.items()}
best_params['meta_model'] = {'alpha': best_alpha}

best_params_path = os.path.join(experiment_dir, "best_parameters.json")
with open(best_params_path, 'w') as f:
    json.dump(best_params, f, indent=4)
print(f"   ✓ Saved: best_parameters.json")

# ----------------------------------------------------------------------------
# 6.4: Save Predictions
# ----------------------------------------------------------------------------

print("\n4. Saving predictions...")

# Out-of-fold predictions
oof_path = os.path.join(experiment_dir, "oof_predictions.npy")
np.save(oof_path, oof_predictions)

# Base model test predictions
test_base_path = os.path.join(experiment_dir, "test_base_predictions.npy")
np.save(test_base_path, test_predictions)

# Final stacked predictions
stacked_path = os.path.join(experiment_dir, "stacked_predictions.npy")
np.save(stacked_path, stacked_results['predictions'])

print(f"   ✓ Saved: oof_predictions.npy")
print(f"   ✓ Saved: test_base_predictions.npy")
print(f"   ✓ Saved: stacked_predictions.npy")

# ----------------------------------------------------------------------------
# 6.5: Save Detailed Results DataFrame
# ----------------------------------------------------------------------------

print("\n5. Saving detailed results dataframe...")

# Create comprehensive results dataframe
results_df = pd.DataFrame({
    'gene_id': df.iloc[test_idx]['gene_id'].values,
    'cell_line': df.iloc[test_idx]['cell_line'].values,
    'actual_expression': y_test.values,
    'stacked_prediction': stacked_results['predictions'],
    'stacked_residual': y_test.values - stacked_results['predictions']
})

# Add individual model predictions
for model_name in tuned_base_models.keys():
    results_df[f'{model_name}_prediction'] = individual_results[model_name]['predictions']
    results_df[f'{model_name}_residual'] = y_test.values - individual_results[model_name]['predictions']

results_path = os.path.join(experiment_dir, "test_predictions_detailed.csv")
results_df.to_csv(results_path, index=False)
print(f"   ✓ Saved: test_predictions_detailed.csv ({len(results_df)} rows)")

# ----------------------------------------------------------------------------
# 6.6: Save Performance Metrics
# ----------------------------------------------------------------------------

print("\n6. Saving performance metrics...")

# Compile all metrics
all_metrics = {
    'stacked_ensemble': {
        'r2': float(stacked_results['r2']),
        'rmse': float(stacked_results['rmse']),
        'mae': float(stacked_results['mae']),
        'pearson': float(stacked_results['pearson']),
        'spearman': float(stacked_results['spearman']),
        'meta_model_alpha': float(stacked_results['alpha']),
        'meta_model_weights': {k: float(v) for k, v in stacked_results['weights'].items()},
        'meta_model_intercept': float(meta_model.intercept_)
    },
    'individual_models': {},
    'improvement': {
        'best_individual_r2': float(best_individual_r2),
        'stacked_r2': float(stacked_results['r2']),
        'improvement_percent': float(improvement)
    }
}

# Add individual model metrics
for model_name, results in individual_results.items():
    all_metrics['individual_models'][model_name] = {
        'r2': float(results['r2']),
        'rmse': float(results['rmse']),
        'mae': float(results['mae']),
        'pearson': float(results['pearson']),
        'spearman': float(results['spearman'])
    }

# Add per-cell-line metrics for stacked ensemble
all_metrics['per_cell_line'] = {}
for cell_line in df['cell_line'].unique():
    mask = results_df['cell_line'] == cell_line
    y_true_cell = results_df.loc[mask, 'actual_expression']
    y_pred_cell = results_df.loc[mask, 'stacked_prediction']
    
    all_metrics['per_cell_line'][cell_line] = {
        'n_samples': int(mask.sum()),
        'r2': float(r2_score(y_true_cell, y_pred_cell)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_cell, y_pred_cell))),
        'mae': float(mean_absolute_error(y_true_cell, y_pred_cell))
    }

metrics_path = os.path.join(experiment_dir, "performance_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=4)
print(f"   ✓ Saved: performance_metrics.json")

# ----------------------------------------------------------------------------
# 6.7: Save Train/Test Split Indices
# ----------------------------------------------------------------------------

print("\n7. Saving train/test split indices...")

split_info = {
    'train_idx': train_idx.tolist(),
    'test_idx': test_idx.tolist(),
    'train_genes': genes_train.unique().tolist(),
    'test_genes': genes_test.unique().tolist()
}

split_path = os.path.join(experiment_dir, "train_test_split.json")
with open(split_path, 'w') as f:
    json.dump(split_info, f, indent=4)
print(f"   ✓ Saved: train_test_split.json")

# ----------------------------------------------------------------------------
# 6.8: Save Experiment Configuration
# ----------------------------------------------------------------------------

print("\n8. Saving experiment configuration...")

config = {
    'experiment_name': experiment_name,
    'timestamp': timestamp,
    'dataset': {
        'source_file': 'models/data/combined_data.csv',
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'n_genes': df['gene_id'].nunique(),
        'cell_lines': df['cell_line'].unique().tolist(),
        'train_test_ratio': f"{80}/{20}"
    },
    'cross_validation': {
        'method': 'GroupKFold',
        'n_folds': n_folds,
        'grouping_variable': 'gene_id'
    },
    'base_models': {
        'count': len(tuned_base_models),
        'names': list(tuned_base_models.keys()),
        'tuning_method': 'RandomizedSearchCV',
        'tuning_iterations': 20
    },
    'meta_model': {
        'type': 'Ridge',
        'alpha': float(best_alpha),
        'tuning_method': 'GridSearchCV'
    },
    'performance_summary': {
        'stacked_r2': float(stacked_results['r2']),
        'stacked_rmse': float(stacked_results['rmse']),
        'best_individual_r2': float(best_individual_r2),
        'improvement_percent': float(improvement)
    }
}

config_path = os.path.join(experiment_dir, "experiment_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
print(f"   ✓ Saved: experiment_config.json")

# ----------------------------------------------------------------------------
# 6.9: Create Performance Summary CSV
# ----------------------------------------------------------------------------

print("\n9. Saving performance summary...")

summary_data = []
for model_name, results in individual_results.items():
    summary_data.append({
        'Model': model_name,
        'Type': 'Individual',
        'R²': results['r2'],
        'RMSE': results['rmse'],
        'MAE': results['mae'],
        'Pearson': results['pearson'],
        'Spearman': results['spearman']
    })

summary_data.append({
    'Model': 'STACKED_ENSEMBLE',
    'Type': 'Ensemble',
    'R²': stacked_results['r2'],
    'RMSE': stacked_results['rmse'],
    'MAE': stacked_results['mae'],
    'Pearson': stacked_results['pearson'],
    'Spearman': stacked_results['spearman']
})

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('R²', ascending=False)

summary_path = os.path.join(experiment_dir, "performance_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"   ✓ Saved: performance_summary.csv")

# ----------------------------------------------------------------------------
# 6.10: Create README
# ----------------------------------------------------------------------------

print("\n10. Creating README...")

readme_content = f"""# Ensemble Stacking Model - {experiment_name}

## Experiment Information
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Stacked Ensemble R²**: {stacked_results['r2']:.4f}
- **Best Individual R²**: {best_individual_r2:.4f}
- **Improvement**: {improvement:+.2f}%

## Dataset Information
- **Source**: models/data/combined_data.csv
- **Total samples**: {len(df):,}
- **Training samples**: {len(X_train):,}
- **Test samples**: {len(X_test):,}
- **Features**: {X.shape[1]:,}
- **Unique genes**: {df['gene_id'].nunique():,}
- **Cell lines**: {', '.join(df['cell_line'].unique())}

## Model Architecture

### Base Models (Level 0)
{chr(10).join([f"{i+1}. {name} - R² = {individual_results[name]['r2']:.4f}" for i, name in enumerate(tuned_base_models.keys())])}

### Meta-Model (Level 1)
- **Type**: Ridge Regression
- **Alpha**: {best_alpha}
- **Performance**: R² = {stacked_results['r2']:.4f}

## Cross-Validation Strategy
- **Method**: GroupKFold (gene-based splitting)
- **Folds**: {n_folds}
- **Grouping**: gene_id (prevents data leakage across cell lines)

## Meta-Model Weights
{chr(10).join([f"- {name}: {stacked_results['weights'][name]:.4f}" for name in stacked_results['weights'].keys()])}

## Performance by Cell Line
{chr(10).join([f"- {cell}: R² = {all_metrics['per_cell_line'][cell]['r2']:.4f}, RMSE = {all_metrics['per_cell_line'][cell]['rmse']:.4f}" for cell in all_metrics['per_cell_line'].keys()])}

## Files in this Directory

### Models
- `base_models/` - Individual tuned base models (5 .pkl files)
- `meta_model.pkl` - Final stacking meta-model

### Predictions
- `oof_predictions.npy` - Out-of-fold predictions from base models (training)
- `test_base_predictions.npy` - Base model predictions on test set
- `stacked_predictions.npy` - Final ensemble predictions on test set
- `test_predictions_detailed.csv` - Complete predictions with all models

### Results & Metrics
- `performance_summary.csv` - Summary table of all model performances
- `performance_metrics.json` - Detailed metrics (overall + per-cell-line)
- `tuning_results.pkl` - Full hyperparameter tuning results

### Configuration
- `experiment_config.json` - Complete experiment setup
- `best_parameters.json` - Optimized hyperparameters for all models
- `train_test_split.json` - Train/test split indices and gene lists

## How to Load and Use Models
```python
import joblib
import numpy as np
import pandas as pd

# Load meta-model
meta_model = joblib.load('meta_model.pkl')

# Load base models
base_models = {{}}
model_names = {list(tuned_base_models.keys())}
for name in model_names:
    base_models[name] = joblib.load(f'base_models/{{name}}.pkl')

# Make predictions on new data
def predict_ensemble(X_new):
    # Get base model predictions
    base_preds = np.zeros((len(X_new), len(base_models)))
    for i, model in enumerate(base_models.values()):
        base_preds[:, i] = model.predict(X_new)
    
    # Get final predictions from meta-model
    final_preds = meta_model.predict(base_preds)
    return final_preds

# Example: predictions = predict_ensemble(X_test)
```

## Notes
- Gene-based splitting ensures no gene appears in both train and test sets
- All models trained with proper regularization to prevent overfitting
- Meta-model weights show contribution of each base model to final predictions

---
*Generated by ensemble stacking pipeline on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

readme_path = os.path.join(experiment_dir, "README.md")
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"   ✓ Saved: README.md")

# ============================================================================
# SAVE SUMMARY
# ============================================================================

print("\n" + "="*60)
print("SAVE SUMMARY")
print("="*60)
print(f"\n📁 All files saved to: {experiment_dir}/")
print(f"\n📊 Files saved:")
print(f"   • Base models: 5 .pkl files in base_models/")
print(f"   • Meta-model: meta_model.pkl")
print(f"   • Predictions: 3 .npy files + 1 detailed .csv")
print(f"   • Performance: performance_summary.csv + performance_metrics.json")
print(f"   • Configuration: 3 .json files")
print(f"   • Tuning results: tuning_results.pkl + best_parameters.json")
print(f"   • Documentation: README.md")
print(f"\n💾 Total: {len(os.listdir(experiment_dir)) + len(os.listdir(base_models_dir)) - 1} files")
print("="*60)

# Print final summary table
print("\n" + "="*70)
print(" " * 20 + "FINAL PERFORMANCE SUMMARY")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)
print(f"\n🏆 Best Model: STACKED_ENSEMBLE (R² = {stacked_results['r2']:.4f})")
print(f"📈 Improvement: {improvement:+.2f}% over best individual model")
print("="*70)

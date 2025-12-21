import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from scipy import stats

# expr 
df = pd.read_csv("models/data/combined_data.csv",
                   header=0, sep=",")
X = df.iloc[:, df.columns.str.contains(r"(_bin)|(cell_line)", regex=True)]
y = df.iloc[:, df.columns.str.contains(r"(cell_line)|(expression)", regex=True)]
histone_mark_num = 5
cell_line_num = 4
bin_len = 200
histone_marks = ['H3K4me3', 'H3K4me1', 'H3K9me3', 'H3K27me3', 'H3K27ac']
cell_lines = ['HeyA8', 'OVCA429', 'PEO1', 'SKOV3']

## Basic Data Quality --------
def check_data_quality(df, id_cols=['gene_id', 'cell_line']):
    """
    Check basic data quality metrics.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to check
    id_cols : list
        Column names that serve as identifiers (default: ['gene_id', 'cell_line'])
    
    Returns:
    --------
    dict : Dictionary containing all quality check results
    """
    results = {}
    
    # 1. Data shape and types
    print("="*50)
    print("BASIC DATA QUALITY CHECKS")
    print("="*50)
    print(f"\nDataset shape: {df.shape}")
    print(f"  - {df.shape[0]:,} rows")
    print(f"  - {df.shape[1]:,} columns")
    
    print("\nColumn types:")
    print(df.dtypes.value_counts())
    results['shape'] = df.shape
    results['dtypes'] = df.dtypes.value_counts()
    
    # 2. Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).query('missing_count > 0')
    
    print("\n" + "-"*50)
    print("Missing values:")
    if len(missing_summary) > 0:
        print(missing_summary)
    else:
        print("✓ No missing values!")
    results['missing'] = missing_summary
    
    # 3. Duplicates
    duplicates = df.duplicated().sum()
    print("\n" + "-"*50)
    print(f"Duplicate rows: {duplicates}")
    results['duplicates'] = duplicates
    
    # Check for duplicate identifier combinations
    if id_cols:
        dup_ids = df.duplicated(subset=id_cols).sum()
        print(f"Duplicate {'+'.join(id_cols)} pairs: {dup_ids}")
        results['duplicate_ids'] = dup_ids
    
    # 4. Check identifier columns
    print("\n" + "-"*50)
    print("Identifier summary:")
    for col in id_cols:
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f"  - Unique {col}: {n_unique:,}")
            results[f'unique_{col}'] = n_unique
            
            # Show distribution for categorical identifiers
            if df[col].dtype == 'object' and n_unique < 20:
                print(f"\n{col} distribution:")
                print(df[col].value_counts())
                results[f'{col}_distribution'] = df[col].value_counts()
    
    print("="*50)
    return results

# Usage
quality_results = check_data_quality(df)

## Target variable analysis ----------

def analyze_target_variable(df, target_col='expression', 
                            group_col='cell_line', figsize=(15, 5),
                            save_outputs=True):
    """
    Analyze target variable distribution and characteristics.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    target_col : str
        Name of target column (default: 'expression')
    group_col : str
        Column to group by for comparison (default: 'cell_line')
    figsize : tuple
        Figure size for plots
    save_outputs : bool
        Whether to save figures and results (default: True)
    
    Returns:
    --------
    dict : Dictionary containing target variable statistics
    """
    # Create directories if they don't exist
    if save_outputs:
        os.makedirs('analysis/QC', exist_ok=True)
        os.makedirs('models/results/QC', exist_ok=True)
    
    results = {}
    
    print("="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    # 1. Basic statistics
    print(f"\nTarget: {target_col}")
    print(df[target_col].describe())
    results['statistics'] = df[target_col].describe()
    
    # 2. Distribution characteristics
    print("\n" + "-"*50)
    print("Distribution characteristics:")
    
    skewness = df[target_col].skew()
    kurtosis = df[target_col].kurtosis()
    excess_kurtosis = kurtosis  # pandas uses excess kurtosis by default
    
    # Skewness interpretation
    if abs(skewness) < 0.5:
        skew_interp = "symmetric"
    elif abs(skewness) < 1:
        skew_interp = "moderate skew"
    elif abs(skewness) < 2:
        skew_interp = "strong skew"
    else:
        skew_interp = "extremely skewed"
    
    direction = "right" if skewness > 0 else "left"
    
    print(f"  - Skewness: {skewness:.3f} → {skew_interp}")
    if abs(skewness) > 0.5:
        print(f"    ({direction} tail dominates)")
    
    # Kurtosis interpretation
    if abs(excess_kurtosis) < 1:
        kurt_interp = "normal tails"
    elif abs(excess_kurtosis) < 3:
        kurt_interp = "heavy tails"
    elif excess_kurtosis >= 5:
        kurt_interp = "extreme outliers dominate"
    else:
        kurt_interp = "moderately heavy tails"
    
    print(f"  - Excess Kurtosis: {excess_kurtosis:.3f} → {kurt_interp}")
    
    results['skewness'] = skewness
    results['skewness_interpretation'] = skew_interp
    results['kurtosis'] = excess_kurtosis
    results['kurtosis_interpretation'] = kurt_interp
    
    # 3. Check for outliers (IQR method)
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    
    print("\n" + "-"*50)
    print("Outliers (IQR method):")
    print(f"  - Lower bound: {lower_bound:.3f}")
    print(f"  - Upper bound: {upper_bound:.3f}")
    print(f"  - Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    results['outliers'] = {'count': len(outliers), 'pct': len(outliers)/len(df)*100}
    
    # 4. Distribution across groups
    if group_col and group_col in df.columns:
        print("\n" + "-"*50)
        print(f"Distribution by {group_col}:")
        group_stats = df.groupby(group_col)[target_col].describe()
        print(group_stats)
        results['group_statistics'] = group_stats
    
    # 5. Visualizations
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Histogram
    axes[0].hist(df[target_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df[target_col].mean(), color='red', 
                    linestyle='--', label=f'Mean: {df[target_col].mean():.2f}')
    axes[0].axvline(df[target_col].median(), color='green', 
                    linestyle='--', label=f'Median: {df[target_col].median():.2f}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution\nSkew: {skewness:.2f} ({skew_interp})')
    axes[0].legend()
    
    # Q-Q plot
    stats.probplot(df[target_col], dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot (Normality Check)\nKurtosis: {excess_kurtosis:.2f} ({kurt_interp})')
    
    # Boxplot by group
    if group_col and group_col in df.columns:
        df.boxplot(column=target_col, by=group_col, ax=axes[2])
        axes[2].set_title(f'{target_col} by {group_col}')
        axes[2].set_xlabel(group_col)
        plt.sca(axes[2])
        plt.xticks(rotation=45)
    else:
        axes[2].boxplot(df[target_col])
        axes[2].set_title(f'{target_col} Boxplot')
    
    plt.tight_layout()
    
    # Save figure
    if save_outputs:
        fig_path = f'analysis/QC/target_distribution_{target_col}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved: {fig_path}")
    
    plt.show()
    
    # Save results to file
    if save_outputs:
        results_path = 'models/results/QC/target_analysis.txt'
        with open(results_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("TARGET VARIABLE ANALYSIS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Target: {target_col}\n\n")
            f.write(str(df[target_col].describe()) + "\n\n")
            f.write("-"*50 + "\n")
            f.write(f"Skewness: {skewness:.3f} → {skew_interp}\n")
            f.write(f"Excess Kurtosis: {excess_kurtosis:.3f} → {kurt_interp}\n\n")
            f.write("-"*50 + "\n")
            f.write(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)\n")
            if group_col and group_col in df.columns:
                f.write("\n" + "-"*50 + "\n")
                f.write(f"Distribution by {group_col}:\n")
                f.write(str(group_stats) + "\n")
        print(f"✓ Results saved: {results_path}")
    
    print("="*50)
    return results

# Usage
target_results = analyze_target_variable(df)

def analyze_feature_characteristics(df, target_col='expression',
                                   id_cols=['gene_id', 'cell_line'],
                                   variance_threshold=0.01,
                                   correlation_threshold=0.95,
                                   sample_features=20,
                                   save_outputs=True):
    """
    Analyze feature characteristics for high-dimensional data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    target_col : str
        Name of target column to exclude from feature analysis
    id_cols : list
        Identifier columns to exclude from feature analysis
    variance_threshold : float
        Threshold for near-zero variance (default: 0.01)
    correlation_threshold : float
        Threshold for high correlation (default: 0.95)
    sample_features : int
        Number of features to sample for detailed visualization
    save_outputs : bool
        Whether to save figures and results
    
    Returns:
    --------
    dict : Dictionary containing feature analysis results
    """
    # Create directories
    if save_outputs:
        os.makedirs('analysis/QC', exist_ok=True)
        os.makedirs('models/results/QC', exist_ok=True)
    
    results = {}
    
    # Get feature columns
    exclude_cols = [target_col] + id_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print("="*50)
    print("FEATURE CHARACTERISTICS ANALYSIS")
    print("="*50)
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Numeric features: {len(numeric_features)}")
    
    # Extract feature groups (e.g., H3K4me3, H3K27ac, etc.)
    feature_groups = {}
    for feat in numeric_features:
        # Extract histone mark name (everything before '_bin')
        if '_bin' in feat:
            group = feat.split('_bin')[0]
            if group not in feature_groups:
                feature_groups[group] = []
            feature_groups[group].append(feat)
    
    print(f"\nFeature groups detected: {list(feature_groups.keys())}")
    for group, feats in feature_groups.items():
        print(f"  - {group}: {len(feats)} features")
    
    results['feature_groups'] = {k: len(v) for k, v in feature_groups.items()}
    
    # 1. VARIANCE ANALYSIS
    print("\n" + "-"*50)
    print("1. VARIANCE ANALYSIS")
    print("-"*50)
    
    variances = df[numeric_features].var()
    
    # Near-zero variance
    low_var_features = variances[variances < variance_threshold]
    print(f"\nNear-zero variance (var < {variance_threshold}):")
    print(f"  - Total: {len(low_var_features)} ({len(low_var_features)/len(numeric_features)*100:.2f}%)")
    
    # Group by histone mark
    if len(low_var_features) > 0:
        print("\n  By histone mark:")
        for group in feature_groups.keys():
            group_low_var = [f for f in low_var_features.index if f.startswith(group)]
            if len(group_low_var) > 0:
                print(f"    - {group}: {len(group_low_var)}")
    
    # Variance summary
    print(f"\nVariance distribution:")
    print(f"  - Mean: {variances.mean():.4f}")
    print(f"  - Median: {variances.median():.4f}")
    print(f"  - Range: [{variances.min():.4f}, {variances.max():.4f}]")
    
    results['variance'] = {
        'low_variance_count': len(low_var_features),
        'low_variance_features': low_var_features.index.tolist(),
        'mean': variances.mean(),
        'median': variances.median()
    }
    
    # 2. SCALE ANALYSIS
    print("\n" + "-"*50)
    print("2. SCALE ANALYSIS")
    print("-"*50)
    
    feature_ranges = df[numeric_features].max() - df[numeric_features].min()
    feature_means = df[numeric_features].mean()
    feature_stds = df[numeric_features].std()
    
    print(f"\nFeature ranges: [{feature_ranges.min():.4f}, {feature_ranges.max():.4f}]")
    print(f"Feature means: [{feature_means.min():.4f}, {feature_means.max():.4f}]")
    print(f"Feature stds: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")
    
    scale_diff = feature_ranges.max() / (feature_ranges.min() + 1e-10)
    
    if scale_diff > 10:
        print(f"\n⚠ Large scale differences (ratio: {scale_diff:.2f})")
        print("  → Recommend standardization before modeling")
    else:
        print(f"\n✓ Features on similar scales (ratio: {scale_diff:.2f})")
    
    results['scale'] = {
        'range_ratio': scale_diff,
        'needs_scaling': scale_diff > 10
    }
    
    # 3. MISSING VALUES PER FEATURE
    print("\n" + "-"*50)
    print("3. MISSING VALUES")
    print("-"*50)
    
    missing_per_feature = df[numeric_features].isnull().sum()
    features_with_missing = missing_per_feature[missing_per_feature > 0]
    
    print(f"\nFeatures with missing values: {len(features_with_missing)}")
    if len(features_with_missing) > 0:
        print(f"  - Max missing: {features_with_missing.max()} ({features_with_missing.max()/len(df)*100:.2f}%)")
        print(f"  - Top 5:")
        print(features_with_missing.nlargest(5))
    else:
        print("  ✓ No missing values in features!")
    
    results['missing'] = {
        'features_with_missing': len(features_with_missing),
        'max_missing_count': features_with_missing.max() if len(features_with_missing) > 0 else 0
    }
    
    print("\n" + "="*50)
    print("Analysis complete! Run next section for correlation analysis.")
    print("="*50)
    
    # Save summary
    if save_outputs:
        with open('models/results/QC/feature_characteristics.txt', 'w') as f:
            f.write("="*50 + "\n")
            f.write("FEATURE CHARACTERISTICS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total features: {len(numeric_features)}\n\n")
            
            f.write("Feature groups:\n")
            for group, count in results['feature_groups'].items():
                f.write(f"  - {group}: {count}\n")
            
            f.write(f"\nLow variance features: {len(low_var_features)}\n")
            f.write(f"Scale ratio: {scale_diff:.2f}\n")
            f.write(f"Scaling needed: {scale_diff > 10}\n")
            
            if len(low_var_features) > 0:
                f.write("\nLow variance features:\n")
                for feat in low_var_features.index:
                    f.write(f"  - {feat}: {variances[feat]:.6f}\n")
        
        print(f"\n✓ Results saved: models/results/QC/feature_characteristics.txt")
    
    return results

# Usage
feature_results = analyze_feature_characteristics(df)

def analyze_feature_correlations(df, target_col='expression',
                                 id_cols=['gene_id', 'cell_line'],
                                 high_corr_threshold=0.95,
                                 top_n_features=20,
                                 save_outputs=True):
    """
    Analyze feature correlations efficiently for high-dimensional data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    target_col : str
        Name of target column
    id_cols : list
        Identifier columns to exclude
    high_corr_threshold : float
        Threshold for identifying highly correlated pairs (default: 0.95)
    top_n_features : int
        Number of top features to show in visualizations
    save_outputs : bool
        Whether to save figures and results
    
    Returns:
    --------
    dict : Dictionary containing correlation analysis results
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create directories
    if save_outputs:
        os.makedirs('analysis/QC', exist_ok=True)
        os.makedirs('models/results/QC', exist_ok=True)
    
    results = {}
    
    # Get feature columns
    exclude_cols = [target_col] + id_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print("="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    print(f"\nAnalyzing {len(numeric_features)} features...")
    
    # Extract feature groups
    feature_groups = {}
    for feat in numeric_features:
        if '_bin' in feat:
            group = feat.split('_bin')[0]
            if group not in feature_groups:
                feature_groups[group] = []
            feature_groups[group].append(feat)
    
    # 1. CORRELATION WITH TARGET
    print("\n" + "-"*50)
    print("1. CORRELATION WITH TARGET")
    print("-"*50)
    
    target_corrs = df[numeric_features].corrwith(df[target_col])
    target_corrs_abs = target_corrs.abs().sort_values(ascending=False)
    
    print(f"\nTop {top_n_features} features correlated with {target_col}:")
    for i, (feat, corr) in enumerate(target_corrs_abs.head(top_n_features).items(), 1):
        print(f"  {i:2d}. {feat:30s}: {target_corrs[feat]:7.4f}")
    
    # Summary by group
    print(f"\nAverage correlation by histone mark:")
    for group in sorted(feature_groups.keys()):
        group_feats = feature_groups[group]
        avg_corr = target_corrs[group_feats].abs().mean()
        max_corr = target_corrs[group_feats].abs().max()
        print(f"  - {group:15s}: mean={avg_corr:.4f}, max={max_corr:.4f}")
    
    results['target_correlation'] = {
        'top_features': target_corrs_abs.head(top_n_features).to_dict(),
        'mean_abs_corr': target_corrs.abs().mean(),
        'max_abs_corr': target_corrs.abs().max()
    }
    
    # 2. MULTICOLLINEARITY (High feature-feature correlation)
    print("\n" + "-"*50)
    print("2. MULTICOLLINEARITY ANALYSIS")
    print("-"*50)
    print("\nCalculating feature-feature correlations...")
    print("(This may take a moment for 1000 features...)")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Find highly correlated pairs (upper triangle only, exclude diagonal)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    print(f"\nHighly correlated pairs (|r| >= {high_corr_threshold}):")
    print(f"  - Total pairs: {len(high_corr_pairs)}")
    
    if len(high_corr_pairs) > 0:
        print(f"\n  Top 10 pairs:")
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', 
                                                                   key=abs, 
                                                                   ascending=False)
        for idx, row in high_corr_df.head(10).iterrows():
            print(f"    {row['feature1']:30s} <-> {row['feature2']:30s}: {row['correlation']:7.4f}")
        
        # Analyze patterns
        print(f"\n  Within same histone mark:")
        within_group = 0
        for pair in high_corr_pairs:
            f1_group = pair['feature1'].split('_bin')[0] if '_bin' in pair['feature1'] else None
            f2_group = pair['feature2'].split('_bin')[0] if '_bin' in pair['feature2'] else None
            if f1_group == f2_group and f1_group is not None:
                within_group += 1
        print(f"    {within_group} pairs ({within_group/len(high_corr_pairs)*100:.1f}%)")
        
    else:
        print("  ✓ No highly correlated feature pairs found!")
    
    results['multicollinearity'] = {
        'high_corr_pairs_count': len(high_corr_pairs),
        'threshold': high_corr_threshold,
        'high_corr_pairs': high_corr_pairs[:50] if len(high_corr_pairs) > 0 else []
    }
    
    # 3. WITHIN-GROUP CORRELATION PATTERNS
    print("\n" + "-"*50)
    print("3. WITHIN-GROUP CORRELATION PATTERNS")
    print("-"*50)
    
    print("\nAverage within-group correlation:")
    for group in sorted(feature_groups.keys()):
        group_feats = feature_groups[group]
        if len(group_feats) > 1:
            group_corr = corr_matrix.loc[group_feats, group_feats]
            # Get upper triangle (exclude diagonal)
            upper_tri = np.triu(group_corr.values, k=1)
            avg_corr = upper_tri[upper_tri != 0].mean()
            print(f"  - {group:15s}: {avg_corr:.4f}")
    
    print("="*50)
    
    # Save results
    if save_outputs:
        with open('models/results/QC/correlation_analysis.txt', 'w') as f:
            f.write("="*50 + "\n")
            f.write("CORRELATION ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Top {top_n_features} features correlated with {target_col}:\n")
            for feat, corr in target_corrs_abs.head(top_n_features).items():
                f.write(f"  {feat}: {target_corrs[feat]:.4f}\n")
            
            f.write(f"\nHighly correlated pairs (|r| >= {high_corr_threshold}): {len(high_corr_pairs)}\n")
            
            if len(high_corr_pairs) > 0:
                f.write("\nTop 50 highly correlated pairs:\n")
                for pair in high_corr_pairs[:50]:
                    f.write(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}\n")
        
        print(f"\n✓ Results saved: models/results/QC/correlation_analysis.txt")
    
    return results

# Usage
correlation_results = analyze_feature_correlations(df)
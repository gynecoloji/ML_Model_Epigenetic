import pandas as pd
import numpy as np

signals = pd.read_csv("features/extracted/chip_signals_per_gene.csv", 
                      header = 0, sep = ",")
signals.index = signals.gene_id.str.replace(r"\.[0-9]*$", "", regex = True).tolist()
signals = signals.drop(columns = ['gene_id'])
signals_filtered = signals.loc[:, signals.columns.str.contains("_bin")]
signals_filtered = np.log2(signals_filtered + 1)
expr = pd.read_csv("data/expression/gene_expression_merged_mean.csv", index_col=0, 
                   header = 0, sep = ",")
expr = expr.groupby(expr.index).mean()

merged = expr.merge(signals_filtered, left_index=True, right_index=True, how='inner')
def reshape_merged_data(merged_df, output_file="models/data/combined_data.csv"):
    """
    Reshape merged dataframe to training format
    
    Input: Wide format (genes × [expressions + features])
    Output: Long format (gene-cellline pairs × [expression + features])
    """
    print("🔄 Reshaping merged data...")
    print(f"  Input shape: {merged_df.shape}")
    
    # Identify cell lines from expression columns
    cell_lines = ['HeyA8', 'OVCA429', 'PEO1', 'SKOV3']  # First 4 columns
    
    # Verify these are the expression columns
    expr_cols = merged_df.columns[:4].tolist()
    print(f"  Expression columns: {expr_cols}")
    
    # Initialize list to store reshaped data
    reshaped_rows = []
    
    # Process each cell line
    for cell_line in cell_lines:
        print(f"\n  Processing {cell_line}...")
        
        # Get expression column for this cell line
        expr_col = cell_line
        
        # Get feature columns for this cell line (all bins)
        feature_cols = [col for col in merged_df.columns 
                       if col.startswith(f"{cell_line}_") and '_bin' in col]
        
        print(f"    - Expression column: {expr_col}")
        print(f"    - Feature columns: {len(feature_cols)}")
        
        # Create dataframe for this cell line
        cell_df = merged_df[[expr_col] + feature_cols].copy()
        
        # Add metadata
        cell_df.insert(0, 'gene_id', merged_df.index)
        cell_df.insert(1, 'cell_line', cell_line)
        
        # Rename expression column to standard name
        cell_df = cell_df.rename(columns={expr_col: 'expression'})
        
        # Rename feature columns to remove cell line prefix (make generic)
        # E.g., HeyA8_H3K4me3_bin1 -> H3K4me3_bin1
        rename_dict = {col: col.replace(f"{cell_line}_", "") 
                      for col in feature_cols}
        cell_df = cell_df.rename(columns=rename_dict)
        
        reshaped_rows.append(cell_df)
        print(f"    ✓ Created {len(cell_df)} rows for {cell_line}")
    
    # Combine all cell lines
    combined = pd.concat(reshaped_rows, ignore_index=True)
    
    print(f"\n  ✓ Combined shape: {combined.shape}")
    print(f"  ✓ Total samples: {len(combined)} ({len(merged_df)} genes × {len(cell_lines)} cell lines)")
    
    # Save
    combined.to_csv(output_file, index=False)
    print(f"  ✓ Saved to: {output_file}")
    
    # Show preview
    print(f"\n📊 Preview:")
    print(combined.head(10))
    
    return combined
merged = reshape_merged_data(merged)
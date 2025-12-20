import pyBigWig
import pandas as pd
import numpy as np
from tqdm import tqdm

def extract_chip_signals_per_bin(genes_bed_file, metadata_file, output_file, n_bins=50):
    """
    Extract ChIP-seq signal values at each bin position along gene bodies
    (strand-aware: bins from 5' to 3' regardless of strand)
    
    Parameters:
    - genes_bed_file: Path to genes_body.bed file
    - metadata_file: Path to sample_metadata.csv
    - output_file: Where to save the extracted signals
    - n_bins: Number of bins to divide each gene into (default=50)
    
    Returns:
    - DataFrame with columns: gene_id, chr, start, end, strand, and
      [CellLine_HistoneMark_bin1, ..., CellLine_HistoneMark_binN] for each sample
      Note: bin1 always represents the 5' end, bin_N the 3' end
    """
    
    # Step 1: Read gene bodies (with strand)
    print("📖 Reading gene body annotations...")
    genes = pd.read_csv(
        genes_bed_file, 
        sep='\t', 
        header=None,
        names=['chr', 'start', 'end', 'gene_id', 'score', 'strand']
    )
    print(f"  ✓ Loaded {len(genes)} genes")
    print(f"  ✓ Strand distribution: {genes['strand'].value_counts().to_dict()}")
    
    # Step 2: Read sample metadata
    print("\n📋 Reading sample metadata...")
    metadata = pd.read_csv(metadata_file)
    print(f"  ✓ Found {len(metadata)} ChIP-seq samples")
    
    # Step 3: Initialize results dataframe (with strand)
    results = pd.DataFrame({
        'gene_id': genes['gene_id'],
        'chr': genes['chr'],
        'start': genes['start'],
        'end': genes['end'],
        'strand': genes['strand']
    })
    
    # Step 4: Extract binned signals for each bigwig file (strand-aware)
    print(f"\n🔬 Extracting ChIP-seq signals ({n_bins} bins per gene, strand-aware)...")
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing samples"):
        sample_id = row['sample_id']
        cell_line = row['cell_line']
        histone_mark = row['histone_mark']
        bw_path = row['bigwig_path']
        
        # Column prefix: CellLine_HistoneMark
        col_prefix = f"{cell_line}_{histone_mark}"
        
        # Open bigwig file
        try:
            bw = pyBigWig.open(bw_path)
        except:
            print(f"⚠️  Warning: Could not open {bw_path}, skipping...")
            continue
        
        # Initialize matrix to store binned signals for all genes
        gene_bin_matrix = np.zeros((len(genes), n_bins))
        
        # Extract binned signals for each gene
        for gene_idx, gene in genes.iterrows():
            try:
                chrom = gene['chr']
                start = int(gene['start'])
                end = int(gene['end'])
                strand = gene['strand']
                
                # Get signal values across the entire region
                values = bw.values(chrom, start, end)
                values = np.array([v if v is not None else 0 for v in values])
                
                if len(values) == 0:
                    gene_bin_matrix[gene_idx, :] = 0
                    continue
                
                # Bin the signal values into n_bins
                bin_size = len(values) / n_bins
                binned_values = []
                
                for bin_idx in range(n_bins):
                    start_idx = int(bin_idx * bin_size)
                    end_idx = int((bin_idx + 1) * bin_size)
                    
                    # Average signal in this bin
                    bin_values = values[start_idx:end_idx]
                    binned_values.append(np.mean(bin_values) if len(bin_values) > 0 else 0.0)
                
                # Reverse bins for negative strand genes (5' to 3' orientation)
                if strand == '-':
                    binned_values = binned_values[::-1]
                
                gene_bin_matrix[gene_idx, :] = binned_values
                
            except Exception as e:
                # If extraction fails, set to 0
                gene_bin_matrix[gene_idx, :] = 0
        
        bw.close()
        
        # Add binned signals as columns to results
        for bin_idx in range(n_bins):
            col_name = f"{col_prefix}_bin{bin_idx+1}"
            results[col_name] = gene_bin_matrix[:, bin_idx]
    
    # Step 5: Save results
    print(f"\n💾 Saving results to: {output_file}")
    results.to_csv(output_file, index=False)
    
    print(f"\n✅ Extraction complete!")
    print(f"   - Genes: {len(results)}")
    print(f"   - Strand distribution: + = {sum(genes['strand'] == '+')}, - = {sum(genes['strand'] == '-')}")
    print(f"   - Bins per gene: {n_bins} (5' → 3' orientation)")
    print(f"   - ChIP samples: {len(metadata)}")
    print(f"   - Total features: {len(metadata) * n_bins} ({len(metadata)} samples × {n_bins} bins)")
    print(f"   - Total columns: {len(results.columns)} (5 metadata + {len(results.columns)-5} features)")
    
    return results

# Run the extraction
signals = extract_chip_signals_per_bin(
    genes_bed_file="data/genes_body_5per_chr1-22_X_protein_coding_width_gt_200.sorted.bed", # 3kb upstream/downstream of gene body boundaries
    metadata_file="features/sample_metadata.csv",
    output_file="features/extracted/chip_signals_per_gene.csv",
    n_bins=200  # You can adjust this (50, 100, etc.)
)

# Preview
print("\n Preview of extracted data (first 5 genes, first 10 columns):")
print(signals.iloc[:5, :10])

# parallel computation
def _process_single_sample(sample_data, genes, n_bins):
    """
    Process a single ChIP-seq sample and extract binned signals
    
    Parameters:
    - sample_data: Dictionary with 'sample_id', 'cell_line', 'histone_mark', 'bigwig_path'
    - genes: DataFrame with gene annotations (chr, start, end, strand)
    - n_bins: Number of bins per gene
    
    Returns:
    - Dictionary with column names as keys and signal arrays as values
    """
    import pyBigWig
    import numpy as np
    
    sample_id = sample_data['sample_id']
    cell_line = sample_data['cell_line']
    histone_mark = sample_data['histone_mark']
    bw_path = sample_data['bigwig_path']
    
    col_prefix = f"{cell_line}_{histone_mark}"
    
    # Open bigwig file
    try:
        bw = pyBigWig.open(bw_path)
    except:
        print(f"⚠️  Warning: Could not open {bw_path}, skipping...")
        return None
    
    # Initialize matrix for this sample
    gene_bin_matrix = np.zeros((len(genes), n_bins))
    
    # Extract binned signals for each gene
    for gene_idx, gene in genes.iterrows():
        try:
            chrom = gene['chr']
            start = int(gene['start'])
            end = int(gene['end'])
            strand = gene['strand']
            
            # Get signal values
            values = bw.values(chrom, start, end)
            values = np.array([v if v is not None else 0 for v in values])
            
            if len(values) == 0:
                gene_bin_matrix[gene_idx, :] = 0
                continue
            
            # Bin the signals
            bin_size = len(values) / n_bins
            binned_values = []
            
            for bin_idx in range(n_bins):
                start_idx = int(bin_idx * bin_size)
                end_idx = int((bin_idx + 1) * bin_size)
                bin_values = values[start_idx:end_idx]
                binned_values.append(np.mean(bin_values) if len(bin_values) > 0 else 0.0)
            
            # Reverse for negative strand
            if strand == '-':
                binned_values = binned_values[::-1]
            
            gene_bin_matrix[gene_idx, :] = binned_values
            
        except Exception as e:
            gene_bin_matrix[gene_idx, :] = 0
    
    bw.close()
    
    # Return results as dictionary: {column_name: values_array}
    result_dict = {}
    for bin_idx in range(n_bins):
        col_name = f"{col_prefix}_bin{bin_idx+1}"
        result_dict[col_name] = gene_bin_matrix[:, bin_idx]
    
    return result_dict
  
  
def extract_chip_signals_per_bin(genes_bed_file, metadata_file, output_file, n_bins=50, n_jobs=-1):
    """
    Extract ChIP-seq signal values at each bin position along gene bodies
    (strand-aware: bins from 5' to 3' regardless of strand)
    PARALLELIZED VERSION
    
    Parameters:
    - genes_bed_file: Path to genes_body.bed file
    - metadata_file: Path to sample_metadata.csv
    - output_file: Where to save the extracted signals
    - n_bins: Number of bins to divide each gene into (default=50)
    - n_jobs: Number of parallel jobs (-1 = all cores, default=-1)
    
    Returns:
    - DataFrame with columns: gene_id, chr, start, end, strand, and
      [CellLine_HistoneMark_bin1, ..., CellLine_HistoneMark_binN] for each sample
    """
    from joblib import Parallel, delayed
    import pandas as pd
    import numpy as np
    
    # Step 1: Read gene bodies
    print("📖 Reading gene body annotations...")
    genes = pd.read_csv(
        genes_bed_file, 
        sep='\t', 
        header=None,
        names=['chr', 'start', 'end', 'gene_id', 'score', 'strand']
    )
    print(f"  ✓ Loaded {len(genes)} genes")
    print(f"  ✓ Strand distribution: {genes['strand'].value_counts().to_dict()}")
    
    # Step 2: Read sample metadata
    print("\n📋 Reading sample metadata...")
    metadata = pd.read_csv(metadata_file)
    print(f"  ✓ Found {len(metadata)} ChIP-seq samples")
    
    # Step 3: Initialize results dataframe
    results = pd.DataFrame({
        'gene_id': genes['gene_id'],
        'chr': genes['chr'],
        'start': genes['start'],
        'end': genes['end'],
        'strand': genes['strand']
    })
    
    # Step 4: Process samples in parallel
    print(f"\n🔬 Extracting ChIP-seq signals ({n_bins} bins per gene, strand-aware)...")
    print(f"⚡ Using parallel processing with {n_jobs if n_jobs > 0 else 'all available'} cores...")
    
    # Convert metadata rows to dictionaries for parallel processing
    sample_data_list = metadata.to_dict('records')
    
    # Process all samples in parallel
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_sample)(sample_data, genes, n_bins) 
        for sample_data in sample_data_list
    )
    
    # Step 5: Merge results from all samples
    print("\n🔄 Merging results from parallel workers...")
    for sample_result in results_list:
        if sample_result is not None:  # Skip failed samples
            for col_name, values in sample_result.items():
                results[col_name] = values
    
    # Step 6: Save results
    print(f"\n💾 Saving results to: {output_file}")
    results.to_csv(output_file, index=False)
    
    print(f"\n✅ Extraction complete!")
    print(f"   - Genes: {len(results)}")
    print(f"   - Strand distribution: + = {sum(genes['strand'] == '+')}, - = {sum(genes['strand'] == '-')}")
    print(f"   - Bins per gene: {n_bins} (5' → 3' orientation)")
    print(f"   - ChIP samples: {len(metadata)}")
    print(f"   - Total features: {len(metadata) * n_bins} ({len(metadata)} samples × {n_bins} bins)")
    print(f"   - Total columns: {len(results.columns)} (5 metadata + {len(results.columns)-5} features)")
    
    return results

results = extract_chip_signals_per_bin(
    genes_bed_file="data/genes_body_5per_chr1-22_X_protein_coding_width_gt_200.sorted.bed",
    metadata_file="features/sample_metadata.csv",
    output_file="features/extracted/chip_signals_per_gene.csv",
    n_bins=200,
    n_jobs=8  # Use 8 cores
)

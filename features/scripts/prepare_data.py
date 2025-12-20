import pandas as pd

def create_sample_metadata(samples_csv="samples.csv", output_file="features/sample_metadata.csv"):
    """
    Create metadata mapping sample_id to cell line and histone mark
    """
    # Read samples
    df = pd.read_csv(samples_csv)
    
    # Filter only ChIP samples (not Input controls)
    chip_samples = df[df['condition'] != 'Input'].copy()
    
    # Extract cell line from notes column
    chip_samples['cell_line'] = chip_samples['notes'].str.extract(r'(PEO1|OVCA429|SKOV3|HeyA8)')
    
    # Create clean metadata
    metadata = chip_samples[['sample_id', 'cell_line', 'condition']].copy()
    metadata.columns = ['sample_id', 'cell_line', 'histone_mark']
    
    # Add bigwig file path
    metadata['bigwig_path'] = 'results/normalized_bigwig/' + metadata['sample_id'] + '.normalized.bw'
    
    # Save
    metadata.to_csv(output_file, index=False)
    print(f"✓ Saved metadata to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Total samples: {len(metadata)}")
    print(f"  - Cell lines: {metadata['cell_line'].unique().tolist()}")
    print(f"  - Histone marks: {metadata['histone_mark'].unique().tolist()}")
    
    return metadata

# Run it
metadata = create_sample_metadata("results/samples.csv")
print("\nFirst few rows:")
print(metadata.head())
import os
from pathlib import Path

def create_project_structure(base_dir="ChIPseq_expression_prediction"):
    """
    Create directory structure for ChIP-seq to expression prediction project
    """
    # Define all directories
    directories = [
        f"{base_dir}/data/expression",
        f"{base_dir}/features/extracted",
        f"{base_dir}/features/scripts",
        f"{base_dir}/models/data",
        f"{base_dir}/models/scripts",
        f"{base_dir}/models/trained",
        f"{base_dir}/models/results",
        f"{base_dir}/analysis/notebooks",
        f"{base_dir}/analysis/figures",
        f"{base_dir}/results/normalized_bigwig"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print(f"\n✓ Project structure created successfully at: {base_dir}/")
    return base_dir

# Run the function
project_dir = create_project_structure("./")
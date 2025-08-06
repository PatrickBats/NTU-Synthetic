#!/usr/bin/env python3
"""
Script to download the Synthetic Konkle dataset from Hugging Face.
This will download and organize all synthetic object images in the same structure
as the original dataset.
"""

import os
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
import argparse

def download_dataset(output_dir: str = "data/SyntheticKonkle", force: bool = False):
    """
    Download the Synthetic Konkle dataset from Hugging Face.
    
    Args:
        output_dir: Directory where the dataset should be saved
        force: If True, will delete existing directory if it exists
    """
    output_path = Path(output_dir)
    
    # Check if directory exists
    if output_path.exists():
        if force:
            print(f"Removing existing directory {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Directory {output_dir} already exists. Use --force to overwrite.")
    
    print(f"Downloading Synthetic Konkle dataset to {output_dir}...")
    
    # Download the dataset
    dataset_path = snapshot_download(
        repo_id="PBATS/SyntheticKonkle",
        local_dir=output_dir,
        repo_type="dataset"
    )
    
    print(f"\nDataset downloaded successfully to {dataset_path}")
    print("The dataset contains synthetic versions of objects rendered in different:")
    print("- Sizes (small, medium, large)")
    print("- Textures (smooth, bumpy)")
    print("- Colors (black, blue, brown, gray, green, orange, pink, purple, red, yellow)")

def main():
    parser = argparse.ArgumentParser(description="Download Synthetic Konkle dataset")
    parser.add_argument("--output", "-o", default="data/SyntheticKonkle",
                      help="Output directory for dataset (default: data/SyntheticKonkle)")
    parser.add_argument("--force", "-f", action="store_true",
                      help="Force download even if directory exists")
    
    args = parser.parse_args()
    download_dataset(args.output, args.force)

if __name__ == "__main__":
    main()
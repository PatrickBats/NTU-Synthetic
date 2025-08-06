#!/usr/bin/env python3
"""
Script to download the Konk Lab Massive Memo dataset
and extract it to the data directory.
"""

import os
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
import argparse

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(str(filename), 'wb') as file, tqdm(
        desc=str(filename),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def download_konklab_dataset(output_dir="data/KonkLab"):
    """
    Download and extract the Konk Lab Massive Memo dataset
    
    Args:
        output_dir: Directory where the dataset should be saved
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download URL
    url = "http://olivalab.mit.edu/MM/archives/ObjectCategories.zip"
    zip_path = output_path / "ObjectCategories.zip"
    
    try:
        print(f"Downloading Konk Lab dataset to {output_dir}...")
        download_file(url, zip_path)
        
        print("\nExtracting files...")
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            zip_ref.extractall(str(output_path))
        
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
        
        # Count extracted files
        file_count = sum(1 for _ in output_path.rglob('*') if _.is_file())
        print(f"\nExtracted {file_count} files successfully!")
        print(f"Dataset is available at: {output_path}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False
        
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file")
        if zip_path.exists():
            zip_path.unlink()
        return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Konk Lab Massive Memo dataset")
    parser.add_argument("--output", "-o", default="data/KonkLab",
                      help="Output directory for dataset (default: data/KonkLab)")
    
    args = parser.parse_args()
    success = download_konklab_dataset(args.output)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
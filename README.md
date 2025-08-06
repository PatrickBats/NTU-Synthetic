# NTU-Synthetic

A research project evaluating the Child's View for Contrastive Learning (CVCL) model's capabilities in fine-grained object classification. CVCL, which learns visual and linguistic associations through a child's perspective, is tested using carefully controlled synthetic datasets.

## Project Overview
This project tests CVCL's ability to perform fine-grained classification by using synthetic objects with controlled variations in:
- Size (small, medium, large)
- Texture (smooth, bumpy)
- Color (10 different colors)

The synthetic dataset allows us to systematically evaluate how well CVCL, trained on a child's natural learning experience, can distinguish subtle differences in object properties.

## Datasets
- **Synthetic Dataset**: Controlled synthetic objects with systematic variations
- **Konk Lab Dataset**: Original object recognition dataset for comparison

## Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ntu-synthetic

# Verify setup
python scripts/test_environment.py

# Download datasets
python scripts/download_synthetic_dataset.py  # For synthetic dataset
python scripts/download_konklab_dataset.py    # For Konk Lab dataset
```

More detailed documentation coming soon.
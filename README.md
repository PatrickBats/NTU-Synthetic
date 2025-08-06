# NTU-Synthetic

A research project evaluating the Child's View for Contrastive Learning (CVCL) model's capabilities in fine-grained object classification. CVCL, which learns visual and linguistic associations through a child's perspective, is tested using carefully controlled synthetic datasets.

## Project Overview
This project tests CVCL's ability to perform fine-grained classification by using synthetic objects with controlled variations in:
- Size (small, medium, large)
- Texture (smooth, bumpy)
- Color (10 different colors)

The synthetic dataset allows us to systematically evaluate how well CVCL, trained on a child's natural learning experience, can distinguish subtle differences in object properties.

## Setup

### 1. Create conda environment
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate ntu-synthetic
```

### 2. Install additional requirements
```bash
# Install spaCy language model
python -m spacy download en_core_web_sm

# Verify setup
python scripts/test_environment.py
```

### 3. Download datasets
```bash
# Download synthetic dataset
python scripts/download_synthetic_dataset.py

# Download Konk Lab dataset
python scripts/download_konklab_dataset.py
```

## Running Classification Tests
The classification tests compare CLIP and CVCL models on various object recognition tasks:

```bash
# Open the classification notebook
jupyter notebook PatrickProject/KonkLab_Classification/run_classification_tests.ipynb
```

## Project Structure
- `data/` - Contains datasets
  - `KonkLab/` - Original Konk Lab dataset
  - `SyntheticKonkle/` - Synthetic dataset with controlled variations
- `scripts/` - Utility scripts for downloading datasets and testing environment
- `src/` - Core model utilities
- `PatrickProject/` - Classification experiments and analysis

## Models Used
- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model
- **CVCL**: Child's View for Contrastive Learning model from [multimodal-baby](https://github.com/wkvong/multimodal-baby)
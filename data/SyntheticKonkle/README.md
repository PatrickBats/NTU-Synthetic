# SyntheticKonkle Dataset

Synthetic object images with controlled variations in color, size, texture, and appearance variants.

## Dataset Structure

### Master File
- **master_labels.csv**: Central metadata for all images
  - Columns: `folder, filename, class, color, size, texture, variant`
  - One row per image across all folders

### Folder Organization
Each object class has two folder types:

#### Color Folders: `{class}_color/`
- **120 images per class** (10 colors × 3 sizes × 2 textures × 2 variants)
- Naming pattern: `{class}_{size}_{texture}_{variant}_{color}.png`
- Example: `apple_large_bumpy_01_red.png`

**Variations:**
- Colors (10): black, blue, brown, gray, green, orange, pink, purple, red, yellow
- Sizes (3): small, medium, large
- Textures (2): smooth, bumpy
- Variants (2): 01, 02

#### Base Folders: `{class}_bases/`
- **12 images per class** (3 sizes × 2 textures × 2 variants)
- Naming pattern: `base_{size}_{texture}_{variant}_{class}.png`
- Example: `base_large_smooth_01_apple.png`
- Natural textures without color modifications

### Per-Folder Labels
Each folder contains its own **labels.csv** with metadata for images in that folder only

## Object Classes
Examples include:
- abacus, apple, axe, babushkadolls, bagel, ball, balloon
- barbiedoll, baseballcards, basket, bathsuit, beanbagchair
- And many more...

## Usage in Project

### Classification Experiments
Used in `PatrickProject/SyntheticKonkle_Classification/` notebooks for:
- Controlled testing of color recognition
- Size discrimination tasks
- Texture classification
- Multi-property combination tests

### Data Loading
```python
import pandas as pd
# Load all metadata
master = pd.read_csv('data/SyntheticKonkle/master_labels.csv')

# Load specific folder metadata
folder_data = pd.read_csv('data/SyntheticKonkle/apple_color/labels.csv')
```

## Key Features
- **Systematic variations**: Every combination of properties exists
- **Controlled testing**: Ideal for isolating specific visual features
- **High image count**: 132 images per object class (120 color + 12 base)
- **Consistent naming**: Predictable file naming convention

## Renaming Objects
Use `universal_rename_script.py` to rename classes:
```python
# Edit lines 21-22
OLD_NAME = "jack-o-lantern"
NEW_NAME = "pumpkin"
```

## Notes
- Synthetic generation allows perfect control over visual properties
- Used to complement real-world KonkLab dataset
- Essential for systematic property-based classification tests
- Check for CSV formatting issues after renaming operations
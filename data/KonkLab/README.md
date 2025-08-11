# KonkLab Dataset

Real-world object images from the Konkle Lab at Harvard, used for testing visual concept recognition.

## Dataset Structure

### Files
- **testdata.csv**: Main metadata file containing all image annotations
  - Columns: `Class, Filename, Color, Size, Amount, Texture`
  - Used by all classification notebooks for loading image properties

- **Konkclasses.csv**: List of all object classes in the dataset

### Image Directory: 17-objects/
Contains 17 object class folders with real photograph images:
- abacus, airplane, apple, armyguy, axe, babushkadolls, babycarriage
- backpack, bagel, ball, balloon, barbiedoll, baseballcards, basket
- bathsuit, beanbagchair, bearteddy

Each class folder contains multiple real-world photographs of that object type with varying:
- Colors (natural variations)
- Sizes (as captured in photos)
- Textures (real-world textures)
- Viewing angles

## Usage in Project

### Classification Tests
Used in `PatrickProject/KonkLab_Classification/` notebooks for:
- Class-level recognition (Class/)
- Color discrimination (Color/)
- Size discrimination (Size/)
- Texture discrimination (Texture/)
- Combined property tests (Color_Size/, Color_Texture/, etc.)

### Loading Data
```python
import pandas as pd
data = pd.read_csv('data/KonkLab/testdata.csv')
# Access images via: data/KonkLab/17-objects/{class}/{filename}
```

## Key Properties
- **Total Classes**: 17 object categories
- **Image Format**: JPG files (various resolutions)
- **Annotations**: Color, Size, Amount, Texture per image
- **Source**: Konkle Lab object dataset from MIT

## Notes
- Real photographs with natural variation
- Multiple exemplars per class
- Used as benchmark for comparing CVCL vs CLIP models
- Column names are case-sensitive (use "Color" not "colour")
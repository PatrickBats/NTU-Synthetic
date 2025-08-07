# NTU Synthetic - Comparing CVCL and CLIP Model Performance

This project evaluates and compares the performance of CVCL (Child Visual Concept Learning) and CLIP models on various classification tasks using the KonkLab dataset.

## Project Structure

- `PatrickProject/`: Main project code
  - `KonkLab_Classification/`: Classification experiments
    - `Class/`: Basic class-level classification tests
    - `Color/`: Color-based classification tests
      - `DifferentClass_DifferentColors/`: Tests across different classes and colors
      - `SameClass_DifferentColors/`: Tests within same class, different colors
    - `Color_Size/`: Combined color and size classification tests
      - `DifferentClass_DifferentColorSize/`: Tests across different classes, colors, and sizes
      - `SameClass_DifferentColorSize/`: Tests within same class, different colors and sizes
  - `Chart_Generation/`: Result visualization and analysis

## Classification Tasks

### Class Classification
- Tests basic object recognition capabilities
- Evaluates if models can identify objects of the same class

### Color Classification
1. Different Class, Different Colors (DCDC)
   - Tests if models can identify objects by color across different classes
   - Example: red apple vs. blue car, green ball, yellow book

2. Same Class, Different Colors (SCDC)
   - Tests color recognition within the same class
   - Example: red apple vs. green apple, yellow apple, blue apple

### Color and Size Classification
1. Different Class, Different Color and Size (DCDCS)
   - Tests recognition when all properties differ
   - Example: big red apple vs. small blue car, medium green ball

2. Same Class, Different Color and Size (SCDS)
   - Tests color and size recognition within same class
   - Example: big red apple vs. small green apple, medium yellow apple

## Setup

### Environment Setup
```bash
conda env create -f environment.yml
conda activate ntu-synthetic
python -m spacy download en_core_web_sm  # Required for CVCL
```

### Data
The project uses the KonkLab dataset, located in:
- `data/KonkLab/17-objects/`: Object images
- `data/KonkLab/testdata.csv`: Image metadata with class, color, and size information

## Running Tests

Each test is available as a Jupyter notebook in its respective directory. To run:

1. Navigate to the test directory
2. Open the comparison notebook (e.g., `Class_Comparison.ipynb`, `DCDC_Comparison.ipynb`)
3. Run all cells to execute both CVCL and CLIP tests

Results are automatically saved to:
`PatrickProject/Chart_Generation/all_prototype_results.csv`
# Benchmarking Attribute Discrimination in Infant-Scale Vision-Language Models

This repository contains the code and experiments for the paper *"Benchmarking Attribute Discrimination in Infant-Scale Vision-Language Models"* by Patrick Batsell (Rice University), Satoshi Tsutsui, and Bihan Wen (Nanyang Technological University).

## Motivation

Human infants learn to recognize objects and their visual attributes — color, size, texture — from remarkably limited experience. The Child Visual Concept Learning (CVCL) model, trained on just ~37,000 infant egocentric video frames (0.01% of CLIP's training data), achieves impressive object-level recognition. But can it also discriminate *attributes* like color and size the way infants do?

Prior evaluations of infant-scale models focused only on **class-level** recognition (e.g., "Is this a ball or a cup?"). This paper asks a deeper question: **Do these models encode the visual properties that distinguish objects within the same category?**

## Key Findings

We find a striking **dissociation** between infant-trained and web-trained models:

- **CVCL (infant-trained)** excels at object class discrimination (~78%) but fails dramatically at color discrimination (~20%, below the 25% chance baseline). It shows a moderate advantage for size discrimination.
- **CLIP and SigLIP (web-trained)** achieve strong performance on both class (~93%) and color (~68%) discrimination, likely because their text supervision provides explicit color labels.
- **DINO (infant vision, no language)** performs similarly to CVCL on class tasks but also struggles with color, suggesting the deficit stems from limited visual experience rather than the language modality alone.
- **Text-vision tests** reveal that web-trained models can ground color words to images, while CVCL's text encoder has almost no color grounding ability — yet CVCL shows surprising size grounding that CLIP lacks.

These results suggest that infant-scale visual experience builds strong **shape-based representations** but is insufficient for learning fine-grained color features. Language supervision at web scale provides color grounding, but size grounding may emerge differently.

## Benchmark Design

We created a controlled synthetic benchmark of **67 object classes x 9 colors x 3 sizes x 2 textures**, generating 7,236 unique images using OmniGen2. Each object is rendered on a plain white background with systematic attribute variations, enabling precise isolation of individual visual properties.

Two complementary test paradigms:
- **Prototype-based (image-only)**: Create a class prototype by averaging image embeddings, then 4-way forced choice via cosine similarity
- **Text-vision (zero-shot)**: Match a text prompt (e.g., "red ball") directly to candidate images via cosine similarity

We also validate on the **KonkLab** dataset of real photographs to confirm that synthetic results generalize.

## Models Compared

| Model | Training Data | Has Text Encoder |
|-------|--------------|-----------------|
| CVCL | ~37K infant egocentric frames (SAYCam) | Yes |
| DINO (Infant) | SAYCam S subset | No |
| DINO (ImageNet) | ImageNet-1K | No |
| CLIP | 400M web image-text pairs | Yes |
| SigLIP | Web-scale image-text pairs | Yes |
| ResNeXt | ImageNet-1K (supervised) | No |

## Repository Structure

```
NTU-Synthetic/
├── experiments/                         # Experiment scripts
│   ├── run_prototype_class.py              # Class discrimination (synthetic)
│   ├── run_prototype_class_konklab.py      # Class discrimination (real images)
│   ├── run_prototype_color.py              # Color discrimination
│   ├── run_prototype_size.py               # Size discrimination
│   ├── run_prototype_texture.py            # Texture discrimination
│   ├── run_textvision_class.py             # Text-vision class test (synthetic)
│   ├── run_textvision_class_konklab.py     # Text-vision class test (real images)
│   ├── run_textvision_color.py             # Text-vision color test
│   ├── run_textvision_size.py              # Text-vision size test
│   ├── run_synonym_sensitivity.py          # Synonym robustness analysis
│   └── Chart_Generation/
│       ├── generate_all_figures.py         # Master figure generation script
│       ├── figures/                        # Output figures
│       └── *.csv                           # Result summary files
├── src/                                    # Core utilities
│   ├── models/feature_extractor.py         # Unified model interface
│   └── utils/model_loader.py              # Model loading
├── data/
│   ├── SyntheticKonkle/                   # Synthetic benchmark images
│   ├── KonkLab/                           # Real photograph dataset
│   └── CVCL_Konkle_Overlap/              # Class overlap mappings
├── discover-hidden-visual-concepts/        # CVPR 2025 submodule
├── Paper/                                  # Manuscript
└── environment.yml                         # Conda environment
```

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate ntu-synthetic
python -m spacy download en_core_web_sm

# Install submodule
pip install -e discover-hidden-visual-concepts/

# Download datasets
python scripts/download_konklab_dataset.py
python scripts/download_synthetic_dataset.py

# Verify setup
python scripts/test_environment.py
```

## Running Experiments

Each experiment script runs independently with configurable models, seeds, and trial counts:

```bash
cd experiments

# Prototype-based tests (all 6 models)
python run_prototype_class.py --num_trials 4000 --seeds 0 1 2
python run_prototype_color.py --num_trials 4000 --seeds 0 1 2
python run_prototype_size.py --num_trials 4000 --seeds 0 1 2
python run_prototype_texture.py --num_trials 4000 --seeds 0 1 2

# Real image validation
python run_prototype_class_konklab.py --num_trials 4000 --seeds 0 1 2

# Text-vision tests (3 text-encoder models)
python run_textvision_class.py --num_trials 4000 --seeds 0 1 2
python run_textvision_color.py --num_trials 4000 --seeds 0 1 2
python run_textvision_size.py --num_trials 4000 --seeds 0 1 2

# Synonym sensitivity analysis
python run_synonym_sensitivity.py
```

Results are saved as CSV summaries in `experiments/Chart_Generation/`.

## Generating Figures

```bash
cd experiments/Chart_Generation
python generate_all_figures.py          # All figures
python generate_all_figures.py --fig 3  # Specific figure only
```

## Citation

If you use this benchmark or codebase, please cite:

```
Batsell, P., Tsutsui, S., & Wen, B. (2025). Benchmarking Attribute Discrimination
in Infant-Scale Vision-Language Models.
```

## Acknowledgments

This work builds on the [Discovering Hidden Visual Concepts Beyond Linguistic Input in Infant Learning](https://arxiv.org/abs/2501.05205) framework (CVPR 2025).

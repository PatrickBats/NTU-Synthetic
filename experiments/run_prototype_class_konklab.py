"""
Class Discrimination Test - Prototype-Based (KonkLab Real Images)
==================================================================

Tests whether models can discriminate between different object classes
using prototype-based 4-way forced choice on KonkLab real image dataset.

Test Design:
1. Filter KonkLab to classes that overlap with SyntheticKonkle smooth test
2. For each trial:
   - Create prototype from multiple images (>=1 per color) of target class
   - Test: query image (target class) vs 3 distractors (different classes)
   - 4-way forced choice based on cosine similarity to prototype
3. Compare with SyntheticKonkle smooth prototype results

Models tested:
- cvcl-resnext: CVCL with ResNeXt backbone
- clip-res: CLIP with ResNet-50 backbone
- siglip: Google SigLIP model
- dino_s_resnext50: DINO self-supervised (SAYCam S)
- resnext: ImageNet-pretrained ResNeXt-50

Usage:
    python run_prototype_class_konklab.py --num_trials 4000 --seeds 0 1 2
    python run_prototype_class_konklab.py --models cvcl-resnext clip-res
"""

import os
import sys
import argparse
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# Path setup
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'discover-hidden-visual-concepts', 'src'))

# Import from submodule
from utils.model_loader import load_model
from models.feature_extractor import FeatureExtractor

# Data paths
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')
KONKLAB_CSV = os.path.join(REPO_ROOT, 'data', 'KonkLab', 'testdata.csv')
SYNTHETIC_CSV = os.path.join(REPO_ROOT, 'data', 'SyntheticKonkle', 'master_labels.csv')
RESULTS_DIR = os.path.join(REPO_ROOT, 'experiments', 'Chart_Generation')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_overlapping_classes():
    """Find classes that exist in both SyntheticKonkle and KonkLab datasets."""
    # Get SyntheticKonkle classes
    synthetic_df = pd.read_csv(SYNTHETIC_CSV)
    synthetic_classes = set(synthetic_df['class'].unique())

    # Get KonkLab classes
    konklab_df = pd.read_csv(KONKLAB_CSV)
    konklab_classes = set(konklab_df['Class'].unique())

    # Handle naming inconsistencies
    name_mapping = {
        'bread': 'breadloaf',
        'muffin': 'muffins',
        'christmastreeornamentball': 'christmastreeornamantball',
        'candle': 'candleholderwithcandle',
        'camera': 'camcorder',
        'pillow': 'cushion',
        'earrings': 'earings',
        'handheldgame': 'gamehandheld',
        'pumpkin': 'jack-o-lantern',
        'saltandpeppershake': 'saltpeppershake',
        'horse': 'toyhorse',
        'rabbit': 'toyrabbit',
        'dumbell': 'exercise_equipment',
    }

    # Map SyntheticKonkle names to KonkLab names
    mapped_synthetic = set()
    for cls in synthetic_classes:
        if cls in name_mapping:
            mapped_synthetic.add(name_mapping[cls])
        else:
            mapped_synthetic.add(cls)

    # Find overlap (returns KonkLab class names)
    overlapping = mapped_synthetic & konklab_classes

    return sorted(overlapping)


class KonkLabImageDataset(Dataset):
    """Dataset class for KonkLab images."""
    def __init__(self, df, data_dir, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['Class'], row['Filename'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, idx


def extract_embeddings(model_name, model, transform, df, data_dir, device, batch_size=32):
    """Extract embeddings for all images in dataframe."""
    extractor = FeatureExtractor(model_name, model, device)
    dataset = KonkLabImageDataset(df, data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc=f"Extracting embeddings"):
            imgs = imgs.to(device)
            embeddings = extractor.get_img_feature(imgs)
            embeddings = extractor.norm_features(embeddings)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def run_prototype_test(model_name, seed, df, embeddings, classes, num_trials=4000, device='cuda'):
    """
    Run prototype-based 4-way forced choice test.

    For each trial:
    1. Select target class
    2. Create prototype from >=1 image per color (excluding query)
    3. Select query from target class
    4. Select 3 distractors from different classes
    5. Compute similarity of all 4 to prototype
    6. Predict: argmax(similarity)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []

    # Group images by class
    class_indices = defaultdict(list)
    for idx, cls in enumerate(df['Class']):
        class_indices[cls].append(idx)

    # Ensure balanced trials across classes
    trials_per_class = num_trials // len(classes)

    for target_class in tqdm(classes, desc=f"Testing {model_name} (seed {seed})"):
        target_indices = class_indices[target_class]

        if len(target_indices) < 2:
            print(f"Warning: Skipping {target_class} - insufficient images")
            continue

        # Get color distribution for this class
        target_df = df.iloc[target_indices]
        colors = target_df['Color'].unique()

        for trial in range(trials_per_class):
            # Step 1: Create prototype (at least 1 per color)
            prototype_indices = []
            for color in colors:
                color_indices = target_df[target_df['Color'] == color].index.tolist()
                if len(color_indices) > 0:
                    sampled_idx = random.choice(color_indices)
                    prototype_indices.append(sampled_idx)

            if len(prototype_indices) == 0:
                continue

            # Step 2: Select query (not in prototype)
            available_query_indices = [idx for idx in target_indices if idx not in prototype_indices]
            if len(available_query_indices) == 0:
                continue

            query_idx = random.choice(available_query_indices)

            # Step 3: Create prototype embedding
            proto_embeddings = embeddings[prototype_indices]
            proto = proto_embeddings.mean(dim=0)
            proto = proto / proto.norm()

            # Step 4: Select 3 distractor classes
            other_classes = [c for c in classes if c != target_class]
            distractor_classes = random.sample(other_classes, min(3, len(other_classes)))

            # Select 1 random image from each distractor class
            distractor_indices = []
            for dist_class in distractor_classes:
                dist_indices = class_indices[dist_class]
                distractor_indices.append(random.choice(dist_indices))

            # Step 5: 4-way forced choice
            candidate_indices = [query_idx] + distractor_indices
            candidate_embeddings = embeddings[candidate_indices]

            # Compute similarities
            similarities = (candidate_embeddings @ proto).cpu().numpy()
            prediction_idx = np.argmax(similarities)

            # Record result
            predicted_class = df.iloc[candidate_indices[prediction_idx]]['Class']
            correct = int(predicted_class == target_class)
            confidence = float(similarities[prediction_idx])

            results.append({
                'trial': len(results),
                'seed': seed,
                'target_class': target_class,
                'query_class': target_class,
                'distractor1_class': distractor_classes[0] if len(distractor_classes) > 0 else None,
                'distractor2_class': distractor_classes[1] if len(distractor_classes) > 1 else None,
                'distractor3_class': distractor_classes[2] if len(distractor_classes) > 2 else None,
                'predicted_class': predicted_class,
                'correct': correct,
                'confidence': confidence,
                'num_prototype_images': len(prototype_indices),
            })

    return pd.DataFrame(results)


def compute_summary(results_df, model_name):
    """Compute summary statistics with confidence intervals."""
    summary = {
        'model': model_name,
        'total_trials': len(results_df),
        'correct': results_df['correct'].sum(),
        'mean_accuracy': results_df['correct'].mean(),
        'std_accuracy': results_df['correct'].std(),
    }

    # Confidence interval (95%)
    n = len(results_df)
    if n > 0:
        ci = 1.96 * summary['std_accuracy'] / np.sqrt(n)
        summary['ci_lower'] = summary['mean_accuracy'] - ci
        summary['ci_upper'] = summary['mean_accuracy'] + ci
    else:
        summary['ci_lower'] = 0
        summary['ci_upper'] = 0

    # Per-class accuracy
    per_class = results_df.groupby('target_class')['correct'].agg(['sum', 'count', 'mean'])
    summary['per_class_accuracy'] = per_class.to_dict()

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run class discrimination prototype test (KonkLab real images)')
    parser.add_argument('--models', nargs='+', default=['cvcl-resnext', 'clip-res', 'siglip', 'dino_s_resnext50', 'dino_resnet50', 'resnext'],
                        help='Models to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Random seeds for statistical confidence')
    parser.add_argument('--num_trials', type=int, default=4000,
                        help='Total number of trials')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding extraction')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--classes_csv', type=str, default=None,
                        help='CSV with Class column to filter classes (e.g., CVCLKonkMatches.csv)')

    args = parser.parse_args()

    print("=" * 80)
    print("Class Discrimination Test - Prototype-Based (KonkLab Real Images)")
    print("=" * 80)

    # Step 1: Load data and filter to target classes
    print("\n1. Loading data...")
    df = pd.read_csv(KONKLAB_CSV)
    print(f"   Total images: {len(df)}")

    if args.classes_csv:
        # Use provided CSV to determine classes
        classes_df = pd.read_csv(args.classes_csv)
        csv_classes = [c.strip() for c in classes_df['Class'].tolist()]
        # Map names from SyntheticKonkle/CVCL convention to KonkLab convention
        name_mapping = {
            'bread': 'breadloaf', 'muffin': 'muffins',
            'christmastreeornamentball': 'christmastreeornamantball',
            'candle': 'candleholderwithcandle', 'camera': 'camcorder',
            'pillow': 'cushion', 'earrings': 'earings',
            'handheldgame': 'gamehandheld', 'pumpkin': 'jack-o-lantern',
            'saltandpeppershake': 'saltpeppershake',
            'horse': 'toyhorse', 'rabbit': 'toyrabbit',
            'dumbell': 'exercise_equipment',
        }
        overlapping_classes = sorted([name_mapping.get(c, c) for c in csv_classes])
        print(f"   Classes from {args.classes_csv}: {len(overlapping_classes)}")
    else:
        # Get classes that overlap with SyntheticKonkle smooth test
        overlapping_classes = get_overlapping_classes()
        print(f"   Classes overlapping with SyntheticKonkle: {len(overlapping_classes)}")

    # Filter to target classes only
    df = df[df['Class'].isin(overlapping_classes)].reset_index(drop=True)
    print(f"   Images in overlapping classes: {len(df)}")

    available_classes = sorted(df['Class'].unique())
    print(f"   Final classes to test: {len(available_classes)}")
    print(f"   Classes: {', '.join(sorted(available_classes))}")

    # Verify each file actually exists
    valid_indices = []
    for idx, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['Class'], row['Filename'])
        if os.path.exists(img_path):
            valid_indices.append(idx)
        else:
            print(f"   Warning: Missing file {img_path}")

    df = df.iloc[valid_indices].reset_index(drop=True)
    print(f"   Final dataset size: {len(df)} images across {len(df['Class'].unique())} classes")

    # Step 2: Run tests for each model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in args.models:
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 80}")

        all_results = []

        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")

            # Load model
            print(f"Loading model...")
            model, transform = load_model(model_name, seed=seed, device=args.device)

            # Extract embeddings
            print(f"Extracting embeddings...")
            embeddings = extract_embeddings(model_name, model, transform, df, DATA_DIR,
                                           args.device, args.batch_size)

            # Run test
            print(f"Running prototype test...")
            results = run_prototype_test(model_name, seed, df, embeddings,
                                        available_classes, args.num_trials, args.device)

            results['model'] = model_name
            all_results.append(results)

            # Print seed results
            accuracy = results['correct'].mean()
            print(f"Seed {seed} Accuracy: {accuracy:.4f} ({results['correct'].sum()}/{len(results)})")

        # Combine results from all seeds
        combined_results = pd.concat(all_results, ignore_index=True)

        # Save detailed results
        results_file = os.path.join(RESULTS_DIR, f'prototype_class_konklab_{model_name}_results_{timestamp}.csv')
        combined_results.to_csv(results_file, index=False)
        print(f"\nSaved detailed results: {results_file}")

        # Compute and save summary
        summary_data = []
        for seed in args.seeds:
            seed_results = combined_results[combined_results['seed'] == seed]
            summary = compute_summary(seed_results, model_name)
            summary['seed'] = seed
            summary_data.append({
                'model': model_name,
                'seed': seed,
                'total_trials': summary['total_trials'],
                'correct': summary['correct'],
                'mean_accuracy': summary['mean_accuracy'],
                'std_accuracy': summary['std_accuracy'],
                'ci_lower': summary['ci_lower'],
                'ci_upper': summary['ci_upper'],
            })

        # Overall summary across seeds
        overall_accuracy = combined_results['correct'].mean()
        overall_std = combined_results.groupby('seed')['correct'].mean().std()
        summary_data.append({
            'model': model_name,
            'seed': 'overall',
            'total_trials': len(combined_results),
            'correct': combined_results['correct'].sum(),
            'mean_accuracy': overall_accuracy,
            'std_accuracy': overall_std,
            'ci_lower': overall_accuracy - 1.96 * overall_std / np.sqrt(len(args.seeds)),
            'ci_upper': overall_accuracy + 1.96 * overall_std / np.sqrt(len(args.seeds)),
        })

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(RESULTS_DIR, f'prototype_class_konklab_{model_name}_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary: {summary_file}")

        print(f"\nOverall Accuracy: {overall_accuracy:.4f} +- {overall_std:.4f}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

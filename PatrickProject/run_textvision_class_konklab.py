"""
Text-Vision Class Discrimination Test (KonkLab Real Images, CVCL-Konkle Overlap)
================================================================================

Zero-shot text-to-image class discrimination on KonkLab real images,
restricted to the 23 classes overlapping CVCL training and Konkle test set
(from CVCLKonkMatches.csv).

Test Design:
1. Load 23 classes from CVCLKonkMatches.csv, map to KonkLab names
2. Filter KonkLab to overlapping classes
3. For each trial:
   - Encode NATURAL class name as text (e.g., "bread" not "breadloaf")
   - Select query image (target class)
   - Select 3 distractors (same color + size, different class)
   - 4-way forced choice based on cosine similarity(text, image)
4. Compare with prototype-based results and synthetic text-vision results

Models tested (text encoders only):
- cvcl-resnext: CVCL with ResNeXt backbone
- clip-res: CLIP with ResNet-50 backbone
- siglip: Google SigLIP model

Usage:
    python run_textvision_class_konklab.py
    python run_textvision_class_konklab.py --models cvcl-resnext clip-res
    python run_textvision_class_konklab.py --num_trials 4000 --seeds 0 1 2
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
CLASSES_CSV = os.path.join(REPO_ROOT, 'data', 'CVCL_Konkle_Overlap', 'CVCLKonkMatches.csv')
RESULTS_DIR = os.path.join(REPO_ROOT, 'PatrickProject', 'Chart_Generation')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Name mapping: natural (SyntheticKonkle/CVCL) → KonkLab
NATURAL_TO_KONKLAB = {
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
# Reverse: KonkLab → natural
KONKLAB_TO_NATURAL = {v: k for k, v in NATURAL_TO_KONKLAB.items()}


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


def extract_image_embeddings(model_name, model, transform, df, data_dir, device, batch_size=32):
    """Extract normalized image embeddings for all images in dataframe."""
    extractor = FeatureExtractor(model_name, model, device)
    dataset = KonkLabImageDataset(df, data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting image embeddings"):
            imgs = imgs.to(device)
            embeddings = extractor.get_img_feature(imgs)
            embeddings = extractor.norm_features(embeddings)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).float()


def extract_text_embeddings(model_name, model, text_labels, device):
    """Extract normalized text embeddings for class labels."""
    extractor = FeatureExtractor(model_name, model, device)
    with torch.no_grad():
        txt_features = extractor.get_txt_feature(list(text_labels))
        txt_features = extractor.norm_features(txt_features)
    return txt_features.cpu().float()


def run_text_vision_test(model_name, seed, df, img_embeddings, txt_embeddings,
                         available_classes, num_trials=4000):
    """
    Run text-vision 4-way forced choice test on KonkLab real images.

    For each trial:
    1. Select target class, encode natural class name as text
    2. Select query image from target class
    3. Select 3 distractors (same color + size, different class)
    4. 4-way forced choice: argmax cosine similarity(text, image)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    class_to_txt_idx = {cls: i for i, cls in enumerate(available_classes)}

    # Group images by class
    class_indices = defaultdict(list)
    for idx in range(len(df)):
        class_indices[df.iloc[idx]['Class']].append(idx)

    # Group by (color, size, class) for controlled distractors
    color_size_class_indices = defaultdict(lambda: defaultdict(list))
    for idx in range(len(df)):
        row = df.iloc[idx]
        key = (row['Color'], row['Size'])
        color_size_class_indices[key][row['Class']].append(idx)

    trials_per_class = num_trials // len(available_classes)

    for target_class in tqdm(available_classes, desc=f"Testing {model_name} (seed {seed})"):
        target_indices = class_indices[target_class]
        if len(target_indices) == 0:
            continue

        txt_idx = class_to_txt_idx[target_class]
        txt_emb = txt_embeddings[txt_idx]

        # Natural name for this class (used in results)
        natural_name = KONKLAB_TO_NATURAL.get(target_class, target_class)

        for trial in range(trials_per_class):
            # Select target image
            query_idx = random.choice(target_indices)
            query_row = df.iloc[query_idx]
            target_color = query_row['Color']
            target_size = query_row['Size']

            # Find distractors with same color + size but different class
            key = (target_color, target_size)
            distractor_pool = {}
            for cls, indices in color_size_class_indices[key].items():
                if cls != target_class:
                    distractor_pool[cls] = indices

            if len(distractor_pool) < 3:
                # Fallback: relax to just different class
                distractor_pool = {}
                for cls, indices in class_indices.items():
                    if cls != target_class:
                        distractor_pool[cls] = indices

            if len(distractor_pool) < 3:
                continue

            distractor_classes = random.sample(list(distractor_pool.keys()), 3)
            distractor_indices = [random.choice(distractor_pool[dc]) for dc in distractor_classes]

            # 4-way forced choice
            candidate_indices = [query_idx] + distractor_indices
            candidate_embeddings = img_embeddings[candidate_indices]

            similarities = (candidate_embeddings @ txt_emb).numpy()
            prediction_idx = np.argmax(similarities)

            predicted_class = df.iloc[candidate_indices[prediction_idx]]['Class']
            correct = int(predicted_class == target_class)
            confidence = float(similarities[prediction_idx])

            results.append({
                'trial': len(results),
                'seed': seed,
                'target_class': target_class,
                'target_natural_name': natural_name,
                'query_color': target_color,
                'query_size': target_size,
                'distractor1_class': distractor_classes[0],
                'distractor2_class': distractor_classes[1],
                'distractor3_class': distractor_classes[2],
                'predicted_class': predicted_class,
                'correct': correct,
                'confidence': confidence,
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

    n = len(results_df)
    if n > 0:
        ci = 1.96 * summary['std_accuracy'] / np.sqrt(n)
        summary['ci_lower'] = summary['mean_accuracy'] - ci
        summary['ci_upper'] = summary['mean_accuracy'] + ci
    else:
        summary['ci_lower'] = 0
        summary['ci_upper'] = 0

    per_class = results_df.groupby('target_class')['correct'].agg(['sum', 'count', 'mean'])
    summary['per_class_accuracy'] = per_class.to_dict()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Text-vision class discrimination test (KonkLab real, CVCL-Konkle overlap)')
    parser.add_argument('--models', nargs='+',
                        default=['cvcl-resnext', 'clip-res', 'siglip'],
                        help='Models to test (must have text encoder)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Random seeds for statistical confidence')
    parser.add_argument('--num_trials', type=int, default=4000,
                        help='Total number of trials')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding extraction')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print("Text-Vision Class Discrimination Test")
    print("KonkLab Real Images | CVCL-Konkle Overlap Classes")
    print("=" * 80)

    # Step 1: Load class list from CVCLKonkMatches.csv and map to KonkLab names
    print("\n1. Loading CVCL-Konkle overlap classes...")
    classes_df = pd.read_csv(CLASSES_CSV)
    csv_classes = [c.strip() for c in classes_df['Class'].tolist()]
    print(f"   Natural class names from CSV: {len(csv_classes)}")

    # Map to KonkLab names
    konklab_target_classes = sorted([NATURAL_TO_KONKLAB.get(c, c) for c in csv_classes])
    print(f"   Mapped KonkLab names: {len(konklab_target_classes)}")

    # Step 2: Load KonkLab data and filter to target classes
    print("\n2. Loading KonkLab data...")
    df = pd.read_csv(KONKLAB_CSV)
    print(f"   Total images: {len(df)}")

    df = df[df['Class'].isin(konklab_target_classes)].reset_index(drop=True)
    print(f"   After filtering to overlap classes: {len(df)}")

    available_classes = sorted(df['Class'].unique())
    print(f"   Available classes: {len(available_classes)}")
    print(f"   KonkLab names: {', '.join(available_classes)}")

    # Show natural name mapping
    print(f"   Text labels (natural names):")
    for cls in available_classes:
        natural = KONKLAB_TO_NATURAL.get(cls, cls)
        if natural != cls:
            print(f"      {cls} -> \"{natural}\"")
        else:
            print(f"      {cls} -> \"{natural}\"")

    # Verify each file exists
    valid_indices = []
    for idx, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['Class'], row['Filename'])
        if os.path.exists(img_path):
            valid_indices.append(idx)
        else:
            print(f"   Warning: Missing file {img_path}")

    df = df.iloc[valid_indices].reset_index(drop=True)
    print(f"   Final dataset size: {len(df)} images across {len(available_classes)} classes")

    # Build text labels list (natural names, aligned with available_classes)
    text_labels = [KONKLAB_TO_NATURAL.get(cls, cls) for cls in available_classes]
    print(f"\n   Text labels for encoding: {text_labels}")

    # Step 3: Run tests for each model
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

            # Extract image embeddings
            print(f"Extracting image embeddings...")
            img_embeddings = extract_image_embeddings(model_name, model, transform, df, DATA_DIR,
                                                     args.device, args.batch_size)

            # Extract text embeddings using natural names
            print(f"Extracting text embeddings for {len(text_labels)} class labels...")
            txt_embeddings = extract_text_embeddings(model_name, model, text_labels, args.device)

            # Run test
            print(f"Running text-vision test...")
            results = run_text_vision_test(model_name, seed, df, img_embeddings, txt_embeddings,
                                           available_classes, args.num_trials)

            results['model'] = model_name
            all_results.append(results)

            # Print seed results
            accuracy = results['correct'].mean()
            print(f"Seed {seed} Accuracy: {accuracy:.4f} ({results['correct'].sum()}/{len(results)})")

        # Combine results from all seeds
        combined_results = pd.concat(all_results, ignore_index=True)

        # Save detailed results
        results_file = os.path.join(RESULTS_DIR, f'textvision_class_konklab_{model_name}_results_{timestamp}.csv')
        combined_results.to_csv(results_file, index=False)
        print(f"\nSaved detailed results: {results_file}")

        # Compute and save summary
        summary_data = []
        for seed in args.seeds:
            seed_results = combined_results[combined_results['seed'] == seed]
            summary = compute_summary(seed_results, model_name)
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
        summary_file = os.path.join(RESULTS_DIR, f'textvision_class_konklab_{model_name}_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary: {summary_file}")

        print(f"\nOverall Accuracy: {overall_accuracy:.4f} +/- {overall_std:.4f}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

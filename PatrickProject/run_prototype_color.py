"""
Prototype Color Discrimination Test (SCDC)
==========================================
Same Class Different Color: Tests whether models can discriminate
between different colors of the same object class using prototype-based
4-way forced choice.

Models tested: cvcl-resnext, clip-res, siglip, dino_s_resnext50, dino_resnet50, resnext

Usage:
    python run_prototype_color.py
    python run_prototype_color.py --models cvcl-resnext clip-res
    python run_prototype_color.py --n_seeds 3 --trials_per_class 500
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

# Import from discover-hidden-visual-concepts package
from utils.model_loader import load_model
from models.feature_extractor import FeatureExtractor

# Data paths
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'SyntheticKonkle_224')
RESULTS_DIR = os.path.join(REPO_ROOT, 'PatrickProject', 'Chart_Generation')
os.makedirs(RESULTS_DIR, exist_ok=True)


class SyntheticImageDataset(Dataset):
    """Dataset class for SyntheticKonkle images."""
    def __init__(self, df, data_dir, transform):
        self.df = df
        self.data_dir = os.path.join(data_dir, 'SyntheticKonkle')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['folder'], row['filename'])
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img), row['class'], row['color'], row['size'], row['texture'], idx
        except:
            img = Image.new('RGB', (224, 224), color='black')
            return self.transform(img), row['class'], row['color'], row['size'], row['texture'], idx


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    classes = [b[1] for b in batch]
    colors = [b[2] for b in batch]
    sizes = [b[3] for b in batch]
    textures = [b[4] for b in batch]
    idxs = [b[5] for b in batch]
    return imgs, classes, colors, sizes, textures, idxs


def build_synthetic_dataset():
    """Combine all labels.csv files from class_color folders."""
    all_data = []
    base_dir = os.path.join(DATA_DIR, 'SyntheticKonkle')

    class_folders = [d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))
                    and d.endswith('_color')]

    for folder in class_folders:
        labels_path = os.path.join(base_dir, folder, 'labels.csv')
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            df['folder'] = folder
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=['class'])
    combined_df['size'] = combined_df['size'].replace('lage', 'large')
    valid_sizes = ['small', 'medium', 'large']
    combined_df = combined_df[combined_df['size'].isin(valid_sizes)]

    print(f"Loaded {len(combined_df)} images from {len(class_folders)} classes")
    return combined_df


def extract_all_embeddings(model_name, seed, device, batch_size=64):
    """Extract embeddings for all images using specified model."""
    model, transform = load_model(model_name, seed=seed, device=device)
    extractor = FeatureExtractor(model_name, model, device)

    df = build_synthetic_dataset()
    ds = SyntheticImageDataset(df, DATA_DIR, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_embs, all_classes, all_colors, all_sizes, all_textures, all_idxs = [], [], [], [], [], []
    with torch.no_grad():
        for imgs, classes, colors, sizes, textures, idxs in tqdm(loader, desc=f"Extracting {model_name}"):
            feats = extractor.get_img_feature(imgs.to(device))
            feats = extractor.norm_features(feats).cpu().float()
            all_embs.append(feats)
            all_classes.extend(classes)
            all_colors.extend(colors)
            all_sizes.extend(sizes)
            all_textures.extend(textures)
            all_idxs.extend(idxs)

    all_embs = torch.cat(all_embs, dim=0)
    return all_embs, all_classes, all_colors, all_sizes, all_textures, all_idxs


def run_scdc_test(all_embs, all_classes, all_colors, all_sizes, all_textures, all_idxs,
                  seed, trials_per_class=500):
    """Run SCDC test: Same Class Different Color."""
    random.seed(seed)

    # Group by class, size, texture (vary color)
    class_size_texture_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for idx, cls, col, size, texture in zip(all_idxs, all_classes, all_colors, all_sizes, all_textures):
        class_size_texture_groups[cls][(size, texture)][col].append(idx)

    unique_classes = list(class_size_texture_groups.keys())
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for target_class in tqdm(unique_classes, desc="SCDC Test"):
        trials_done = 0

        for (size, texture), color_groups in class_size_texture_groups[target_class].items():
            if trials_done >= trials_per_class:
                break

            available_colors = list(color_groups.keys())
            if len(available_colors) < 4:
                continue

            n_trials = min(50, trials_per_class - trials_done)

            for _ in range(n_trials):
                selected_colors = random.sample(available_colors, 4)
                target_color = selected_colors[0]
                distractor_colors = selected_colors[1:4]

                q = random.choice(color_groups[target_color])
                same_color_group = [i for i in color_groups[target_color] if i != q]

                if same_color_group:
                    proto = all_embs[[all_idxs.index(i) for i in same_color_group]].mean(0)
                else:
                    proto = all_embs[all_idxs.index(q)]
                proto = proto / proto.norm()

                distractors = [random.choice(color_groups[c]) for c in distractor_colors if color_groups[c]]
                if len(distractors) < 3:
                    continue

                candidates = [q] + distractors
                feats_cand = all_embs[[all_idxs.index(i) for i in candidates]]
                sims = feats_cand @ proto
                guess = candidates[sims.argmax().item()]

                class_correct[target_class] += int(guess == q)
                class_total[target_class] += 1
                trials_done += 1

    return {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in unique_classes}


def save_results(model_name, all_results, trials_per_class):
    """Save results to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Detailed results
    detailed_rows = []
    for cls, accs in all_results.items():
        for seed_idx, acc in enumerate(accs):
            detailed_rows.append({
                'model': model_name,
                'class': cls,
                'seed': seed_idx,
                'accuracy': acc,
                'n_trials': trials_per_class,
                'test_type': 'SCDC'
            })

    if detailed_rows:
        df = pd.DataFrame(detailed_rows)
        filename = f'prototype_color_{model_name}_results_{timestamp}.csv'
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        print(f"Saved: {filename}")

    # Summary statistics
    summary_rows = []
    for cls, accs in all_results.items():
        if accs:
            summary_rows.append({
                'model': model_name,
                'class': cls,
                'mean_accuracy': np.mean(accs),
                'std': np.std(accs, ddof=1) if len(accs) > 1 else 0,
                'n_seeds': len(accs),
                'test_type': 'SCDC'
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        filename = f'prototype_color_{model_name}_summary_{timestamp}.csv'
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run prototype color discrimination (SCDC) test')
    parser.add_argument('--models', nargs='+',
                       default=['cvcl-resnext', 'clip-res', 'siglip', 'dino_s_resnext50', 'dino_resnet50', 'resnext'],
                       help='Models to test')
    parser.add_argument('--n_seeds', type=int, default=3,
                       help='Number of random seeds')
    parser.add_argument('--trials_per_class', type=int, default=500,
                       help='Trials per class per seed')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Trials per class: {args.trials_per_class}")

    for model_name in args.models:
        try:
            print(f"\n{'='*60}")
            print(f"SCDC (Color Discrimination): {model_name}")
            print(f"{'='*60}")

            all_results = defaultdict(list)

            for seed in range(args.n_seeds):
                print(f"\n--- Seed {seed + 1}/{args.n_seeds} ---")
                try:
                    embs, classes, colors, sizes, textures, idxs = extract_all_embeddings(
                        model_name, seed=seed, device=args.device
                    )
                except Exception as e:
                    if seed > 0 and "404" in str(e):
                        print(f"Model {model_name} likely doesn't have seed {seed}, skipping...")
                        continue
                    raise e

                class_accs = run_scdc_test(embs, classes, colors, sizes, textures, idxs,
                                           seed=seed, trials_per_class=args.trials_per_class)

                for cls, acc in class_accs.items():
                    all_results[cls].append(acc)

                mean_acc = np.mean(list(class_accs.values())) if class_accs else 0
                print(f"  SCDC mean accuracy: {mean_acc:.3f}")

            save_results(model_name, all_results, args.trials_per_class)

            # Print summary
            all_means = [np.mean(accs) for accs in all_results.values() if accs]
            if all_means:
                print(f"\nOverall SCDC: {np.mean(all_means):.3f}")

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

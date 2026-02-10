"""
Text-Vision Color Discrimination Test
======================================
Query: TEXT prompt (e.g., "red ball")
Candidates: 4 IMAGES - same class, same size, same texture, DIFFERENT colors
Task: Match text to correct image (4-way forced choice, chance = 25%)

Models tested (text encoders only): cvcl-resnext, clip-res, siglip

Usage:
    python run_textvision_color.py
    python run_textvision_color.py --models cvcl-resnext clip-res
    python run_textvision_color.py --n_seeds 3 --trials_per_class 500
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
RESULTS_DIR = os.path.join(REPO_ROOT, 'experiments', 'Chart_Generation')
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
    return model, extractor, all_embs, all_classes, all_colors, all_sizes, all_textures, all_idxs


def run_textvision_color_test(model_name, extractor, all_embs, all_classes, all_colors,
                               all_sizes, all_textures, all_idxs, seed, trials_per_class=500):
    """
    Text-Vision Color Test:
    - Query: TEXT prompt (e.g., "red ball")
    - Candidates: 4 IMAGES - same class, same size, same texture, DIFFERENT colors
    - Task: Match text to correct image (4-way forced choice, chance = 25%)
    """
    random.seed(seed)

    # Group by class -> (size, texture) -> color -> [image indices]
    class_size_texture_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for idx, cls, col, size, texture in zip(all_idxs, all_classes, all_colors, all_sizes, all_textures):
        class_size_texture_groups[cls][(size, texture)][col].append(idx)

    unique_classes = list(class_size_texture_groups.keys())
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for target_class in tqdm(unique_classes, desc=f"TextVision Color ({model_name})"):
        trials_done = 0

        for (size, texture), color_groups in class_size_texture_groups[target_class].items():
            if trials_done >= trials_per_class:
                break

            available_colors = list(color_groups.keys())
            if len(available_colors) < 4:
                continue

            n_trials = min(50, trials_per_class - trials_done)

            for _ in range(n_trials):
                # Pick 4 different colors
                selected_colors = random.sample(available_colors, 4)
                target_color = selected_colors[0]
                distractor_colors = selected_colors[1:4]

                # Get one image per color (all same class, size, texture)
                target_idx = random.choice(color_groups[target_color])
                distractor_idxs = [random.choice(color_groups[c]) for c in distractor_colors]

                # Build 4 image candidates - position 0 is target
                candidate_idxs = [target_idx] + distractor_idxs

                # Get image embeddings
                img_embs = torch.stack([all_embs[all_idxs.index(idx)] for idx in candidate_idxs])

                # Create TEXT QUERY: "{color} {class}"
                text_query = f"{target_color} {target_class}"

                with torch.no_grad():
                    text_feat = extractor.get_txt_feature([text_query])
                    text_feat = extractor.norm_features(text_feat).float()

                img_embs = img_embs.to(text_feat.device).float()

                # Compute similarity: text vs 4 images
                sims = text_feat @ img_embs.T  # [1, 4]
                pred_idx = sims.argmax().item()

                class_correct[target_class] += int(pred_idx == 0)
                class_total[target_class] += 1
                trials_done += 1

    return {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in unique_classes}


def save_results(model_name, all_results, trials_per_class):
    """Save results to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_rows = []
    for cls, accs in all_results.items():
        for seed_idx, acc in enumerate(accs):
            detailed_rows.append({
                'model': model_name,
                'class': cls,
                'seed': seed_idx,
                'accuracy': acc,
                'n_trials': trials_per_class,
                'test_type': 'TV_Color'
            })

    if detailed_rows:
        df = pd.DataFrame(detailed_rows)
        filename = f'textvision_color_{model_name}_results_{timestamp}.csv'
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        print(f"Saved: {filename}")

    summary_rows = []
    for cls, accs in all_results.items():
        if accs:
            summary_rows.append({
                'model': model_name,
                'class': cls,
                'mean_accuracy': np.mean(accs),
                'std': np.std(accs, ddof=1) if len(accs) > 1 else 0,
                'n_seeds': len(accs),
                'test_type': 'TV_Color'
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        filename = f'textvision_color_{model_name}_summary_{timestamp}.csv'
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run text-vision color discrimination test')
    parser.add_argument('--models', nargs='+',
                       default=['cvcl-resnext', 'clip-res', 'siglip'],
                       help='Models to test (must have text encoders)')
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

    print("\n" + "="*60)
    print("TEXT-VISION COLOR TEST METHODOLOGY:")
    print("  Query: TEXT prompt (e.g., 'red ball')")
    print("  Candidates: 4 IMAGES (same class, size, texture; different colors)")
    print("  Task: Match text to correct image")
    print("  Chance: 25% (4-way forced choice)")
    print("="*60)

    for model_name in args.models:
        try:
            all_results = defaultdict(list)

            for seed in range(args.n_seeds):
                print(f"\n--- Seed {seed + 1}/{args.n_seeds} ---")

                try:
                    model, extractor, embs, classes, colors, sizes, textures, idxs = extract_all_embeddings(
                        model_name, seed=seed, device=args.device
                    )
                except Exception as e:
                    if seed > 0:
                        print(f"Model {model_name} likely doesn't have seed {seed}, skipping...")
                        continue
                    raise e

                color_accs = run_textvision_color_test(
                    model_name, extractor, embs, classes, colors, sizes, textures, idxs,
                    seed=seed, trials_per_class=args.trials_per_class
                )
                for cls, acc in color_accs.items():
                    all_results[cls].append(acc)
                mean_color = np.mean(list(color_accs.values())) if color_accs else 0
                print(f"  TV_Color mean accuracy: {mean_color:.3f}")

            save_results(model_name, all_results, args.trials_per_class)

            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {model_name}")
            print(f"{'='*60}")
            all_means = [np.mean(accs) for accs in all_results.values() if accs]
            if all_means:
                print(f"  TV_Color: {np.mean(all_means):.3f}")

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

"""
Text-Vision Size Test - CORRECT IMPLEMENTATION
==============================================

The correct methodology for text-vision size discrimination:
- Query: TEXT prompt (e.g., "small ball")
- Candidates: 4 IMAGES of the same class, same color, same texture, but DIFFERENT sizes
- Task: Match the text prompt to the correct image

This ensures:
1. All 4 image candidates share the same visual properties (color, texture, class)
2. Only SIZE differs between the 4 candidates
3. The model must use size information from the text to select the correct image
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

# Attribute values
SIZES = ['small', 'medium', 'large']


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

    # Fix typo in size column
    combined_df['size'] = combined_df['size'].replace('lage', 'large')
    valid_sizes = ['small', 'medium', 'large']
    combined_df = combined_df[combined_df['size'].isin(valid_sizes)]

    print(f"Loaded {len(combined_df)} images from {len(class_folders)} classes")
    return combined_df


def run_textvision_size_test_correct(model_name, extractor, all_embs, all_classes, all_colors,
                                      all_sizes, all_textures, all_idxs, seed, trials_per_class=500):
    """
    CORRECT Text-Vision Size Test:
    - Query: TEXT prompt (e.g., "small ball")
    - Candidates: 4 IMAGES - same class, same color, same texture, DIFFERENT sizes
    - Task: Match text to correct image (4-way forced choice, chance = 25%)
    """
    random.seed(seed)

    # Group by class, color, texture (vary size) - same as prototype SCDS
    # Structure: class -> (color, texture) -> size -> [image indices]
    class_color_texture_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for idx, cls, col, size, texture in zip(all_idxs, all_classes, all_colors, all_sizes, all_textures):
        class_color_texture_groups[cls][(col, texture)][size].append(idx)

    unique_classes = list(class_color_texture_groups.keys())
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for target_class in tqdm(unique_classes, desc=f"TextVision Size CORRECT ({model_name})"):
        trials_done = 0

        for (col, texture), size_groups in class_color_texture_groups[target_class].items():
            if trials_done >= trials_per_class:
                break

            available_sizes = [s for s in size_groups.keys() if s in SIZES]
            # Need at least 3 sizes for meaningful test
            if len(available_sizes) < 3:
                continue

            n_trials = min(50, trials_per_class - trials_done)

            for _ in range(n_trials):
                # Pick target size (this determines the text query)
                target_size = random.choice(available_sizes)

                # Get distractor sizes
                distractor_sizes = [s for s in available_sizes if s != target_size]

                # For 4-way, we need 3 distractors. If only 2 different distractor sizes,
                # duplicate one of them
                if len(distractor_sizes) == 2:
                    distractor_sizes.append(random.choice(distractor_sizes))

                # BUILD 4 IMAGE CANDIDATES - all same class, same color, same texture, different sizes
                # Position 0 = target (correct answer)
                candidate_sizes = [target_size] + distractor_sizes[:3]

                # Get one image for each size (all from same color/texture group!)
                candidate_images = []
                for size in candidate_sizes:
                    if size in size_groups and len(size_groups[size]) > 0:
                        img_idx = random.choice(size_groups[size])
                        candidate_images.append(img_idx)
                    else:
                        # Skip this trial if we can't find an image for this size
                        break

                if len(candidate_images) != 4:
                    continue  # Skip if we couldn't get 4 images

                # Get image embeddings for all 4 candidates
                img_embs = torch.stack([all_embs[all_idxs.index(idx)] for idx in candidate_images])

                # Create TEXT QUERY: "{size} {class}"
                text_query = f"{target_size} {target_class}"

                with torch.no_grad():
                    text_feat = extractor.get_txt_feature([text_query])
                    text_feat = extractor.norm_features(text_feat).float()

                # Move image embeddings to same device
                img_embs = img_embs.to(text_feat.device).float()

                # Compute similarity: text_query vs all 4 images
                # text_feat shape: [1, D], img_embs shape: [4, D]
                # Result: [1, 4] similarities
                sims = text_feat @ img_embs.T  # [1, 4]

                # Predict image (index 0 is correct answer - the target size image)
                pred_idx = sims.argmax().item()

                class_correct[target_class] += int(pred_idx == 0)
                class_total[target_class] += 1
                trials_done += 1

    return {cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in unique_classes}


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


def run_all_tests_for_model(model_name, n_seeds=3, trials_per_class=500, device='cuda'):
    """Run CORRECT text-vision size test for a single model across multiple seeds."""
    print(f"\n{'='*60}")
    print(f"TEXT-VISION SIZE TEST (CORRECT): {model_name}")
    print(f"Query: TEXT prompt, Candidates: 4 IMAGES (same color/texture)")
    print(f"{'='*60}")

    all_results = defaultdict(list)

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        try:
            model, extractor, embs, classes, colors, sizes, textures, idxs = extract_all_embeddings(
                model_name, seed=seed, device=device
            )
        except Exception as e:
            print(f"Error loading model {model_name} with seed {seed}: {e}")
            if seed > 0:
                print(f"Model {model_name} likely doesn't have seed {seed}, skipping...")
                continue
            raise e

        # Run CORRECT size test
        print("Running CORRECT Text-Vision Size test (text query → image choices)...")
        size_accs = run_textvision_size_test_correct(
            model_name, extractor, embs, classes, colors, sizes, textures, idxs,
            seed=seed, trials_per_class=trials_per_class
        )
        for cls, acc in size_accs.items():
            all_results[cls].append(acc)
        mean_size = np.mean(list(size_accs.values())) if size_accs else 0
        print(f"  TV_Size (correct) mean accuracy: {mean_size:.3f}")

    return all_results


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
                'test_type': 'TV_Size_Correct'
            })

    if detailed_rows:
        df = pd.DataFrame(detailed_rows)
        filename = f'textvision_size_{model_name}_results_{timestamp}.csv'
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
                'test_type': 'TV_Size_Correct'
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        filename = f'textvision_size_{model_name}_summary_{timestamp}.csv'
        df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
        print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run CORRECT text-vision size experiment')
    parser.add_argument('--models', nargs='+',
                       default=['siglip'],
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
    print("CORRECT TEXT-VISION SIZE TEST METHODOLOGY:")
    print("  Query: TEXT prompt (e.g., 'small ball')")
    print("  Candidates: 4 IMAGES (same class, color, texture; different sizes)")
    print("  Task: Match text to correct image")
    print("  Chance: 25% (4-way forced choice)")
    print("="*60)

    for model_name in args.models:
        try:
            results = run_all_tests_for_model(
                model_name,
                n_seeds=args.n_seeds,
                trials_per_class=args.trials_per_class,
                device=args.device
            )
            save_results(model_name, results, args.trials_per_class)

            # Print summary
            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {model_name}")
            print(f"{'='*60}")
            all_means = [np.mean(accs) for accs in results.values() if accs]
            if all_means:
                overall = np.mean(all_means)
                print(f"  TV_Size (correct): {overall:.3f}")

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

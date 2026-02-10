"""
Synonym Sensitivity Test for Appendix
4-way forced choice text-vision test with size synonyms.

Tests whether models understand synonyms for size words:
- small synonyms: little, tiny
- large synonyms: big, huge

Note: "medium" has no synonyms in SAYCam vocabulary, so we skip it
and only test small vs large discrimination with synonym substitutions.
"""

import os
import sys
import random
from collections import defaultdict
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'discover-hidden-visual-concepts', 'src'))

from src.utils.model_loader import load_model
from src.models.feature_extractor import FeatureExtractor

# Paths
DATA_ROOT = os.path.join(REPO_ROOT, 'data', 'SyntheticKonkle')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'PatrickProject', 'Chart_Generation')

# Synonym tests to run
# Each test: (synonym_word, target_size, distractor_size)
SYNONYM_TESTS = [
    ('little', 'small', 'large'),  # small synonym
    ('tiny', 'small', 'large'),    # small synonym
    ('big', 'large', 'small'),     # large synonym
    ('huge', 'large', 'small'),    # large synonym
]

# Models to test
MODELS = ['cvcl-resnext', 'clip-res', 'siglip']

# Test parameters
N_TRIAL_SEEDS = 3  # Random trial sampling seeds (same model, different trials)
MODEL_SEED = 0     # Fixed model seed (for CVCL checkpoint selection)
TRIALS_PER_CLASS = 500


def load_dataset():
    """Load SyntheticKonkle dataset with all sizes (medium needed as distractors)."""
    labels_path = os.path.join(DATA_ROOT, 'master_labels.csv')
    df = pd.read_csv(labels_path)

    # Build full path from folder and filename
    df['full_path'] = df.apply(
        lambda row: os.path.join(DATA_ROOT, row['folder'], row['filename']),
        axis=1
    )

    # Keep all sizes - medium is needed as distractors even though we don't test synonyms for it
    n_classes = df['class'].nunique()
    print(f"Loaded {len(df)} images from {n_classes} classes")
    return df


def extract_embeddings(model_name, seed, df, device, batch_size=64):
    """Extract image embeddings for all images."""
    model, transform = load_model(model_name, seed=seed, device=device)
    extractor = FeatureExtractor(model_name, model, device)

    all_embs = []
    valid_indices = []
    all_paths = df['full_path'].tolist()

    for i in tqdm(range(0, len(all_paths), batch_size), desc=f"Extracting {model_name}"):
        batch_paths = all_paths[i:i+batch_size]
        images = []
        batch_valid_indices = []

        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                img = transform(img)
                images.append(img)
                batch_valid_indices.append(i + j)
            except Exception as e:
                print(f"Skipping corrupted image: {path}")
                continue

        if len(images) == 0:
            continue

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            embs = extractor.get_img_feature(batch)
            embs = extractor.norm_features(embs).float()
        all_embs.append(embs.cpu())
        valid_indices.extend(batch_valid_indices)

    all_embs = torch.cat(all_embs, dim=0)

    # Filter df to only valid indices
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    return all_embs, extractor, df_valid


def run_synonym_test(extractor, all_embs, df, seed, synonym, target_size, distractor_size,
                     trials_per_class=500):
    """
    Run 4-way forced choice test with a specific synonym.

    For each trial:
    - Pick a target image of target_size
    - Pick 3 distractor images of NOT target_size (same class, color, texture)
    - Create text prompt: "{synonym} {class}"
    - Match text to correct image (the target)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group images by class, color, texture, size
    # Key: (class, color, texture) -> {size: [indices]}
    groups = defaultdict(lambda: defaultdict(list))

    for idx, row in df.iterrows():
        key = (row['class'], row['color'], row['texture'])
        groups[key][row['size']].append(idx)

    correct = 0
    total = 0

    # Get unique classes
    unique_classes = df['class'].unique()

    for target_class in unique_classes:
        trials_done = 0

        # Find all (color, texture) combinations for this class
        class_groups = {k: v for k, v in groups.items() if k[0] == target_class}

        for (cls, color, texture), size_dict in class_groups.items():
            if trials_done >= trials_per_class:
                break

            # Need target_size images
            if target_size not in size_dict:
                continue

            target_indices = size_dict[target_size]

            # Distractors can be ANY size except target_size (includes medium)
            distractor_indices = []
            for size, indices in size_dict.items():
                if size != target_size:
                    distractor_indices.extend(indices)

            # Need at least 1 target and 3 distractors
            if len(target_indices) < 1 or len(distractor_indices) < 3:
                continue

            # Run trials for this combination
            n_trials = min(50, trials_per_class - trials_done)

            for _ in range(n_trials):
                # Select 1 target
                target_idx = random.choice(target_indices)

                # Select 3 distractors (without replacement if possible)
                if len(distractor_indices) >= 3:
                    distractor_idxs = random.sample(distractor_indices, 3)
                else:
                    # Sample with replacement if needed
                    distractor_idxs = random.choices(distractor_indices, k=3)

                # Build candidate list: target at random position
                candidates = distractor_idxs.copy()
                target_position = random.randint(0, 3)
                candidates.insert(target_position, target_idx)

                # Get image embeddings for candidates
                cand_embs = torch.stack([all_embs[idx] for idx in candidates])

                # Create text prompt with synonym
                text_prompt = f"{synonym} {target_class}"

                # Get text embedding
                with torch.no_grad():
                    text_feat = extractor.get_txt_feature([text_prompt])
                    text_feat = extractor.norm_features(text_feat).float().squeeze(0).cpu()

                # Compute similarities (both on CPU)
                sims = cand_embs @ text_feat
                pred_position = sims.argmax().item()

                # Check if correct
                if pred_position == target_position:
                    correct += 1
                total += 1
                trials_done += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    df = load_dataset()

    # Results storage
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print('='*60)

        # Extract embeddings once per model (using fixed MODEL_SEED)
        all_embs, extractor, df_valid = extract_embeddings(model_name, MODEL_SEED, df, device)

        # Run 3 different random trial samplings
        for trial_seed in range(N_TRIAL_SEEDS):
            print(f"\nTrial seed {trial_seed}")

            # Run each synonym test
            for synonym, target_size, distractor_size in SYNONYM_TESTS:
                acc, n_trials = run_synonym_test(
                    extractor, all_embs, df_valid, trial_seed,
                    synonym, target_size, distractor_size,
                    TRIALS_PER_CLASS
                )

                results.append({
                    'model': model_name,
                    'trial_seed': trial_seed,
                    'synonym': synonym,
                    'target_size': target_size,
                    'accuracy': acc,
                    'n_trials': n_trials
                })

                print(f"  {synonym} ({target_size}): {acc*100:.1f}% ({n_trials} trials)")

        # Clean up after all trial seeds for this model
        del all_embs, extractor
        torch.cuda.empty_cache()

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, f'synonym_sensitivity_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY: Mean accuracy across seeds (chance = 25%)")
    print("="*60)

    summary = results_df.groupby(['model', 'synonym', 'target_size'])['accuracy'].agg(['mean', 'std']).reset_index()
    summary['std'] = summary['std'].fillna(0)  # Handle case with single trial seed
    summary['accuracy_pct'] = summary['mean'] * 100
    summary['std_pct'] = summary['std'] * 100

    for model in MODELS:
        print(f"\n{model}:")
        model_data = summary[summary['model'] == model]
        for _, row in model_data.iterrows():
            print(f"  {row['synonym']:6s} ({row['target_size']:5s}): {row['accuracy_pct']:.1f}% (±{row['std_pct']:.1f})")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f'synonym_sensitivity_summary_{timestamp}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Compare to baseline (small/large)
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("Compare these results to baseline text-vision size test:")
    print("- If synonym accuracy ≈ baseline: model understands synonyms")
    print("- If synonym accuracy < baseline: model struggles with that word")
    print("\nNote: 'medium' excluded because no synonyms exist in SAYCam vocabulary")


if __name__ == '__main__':
    main()

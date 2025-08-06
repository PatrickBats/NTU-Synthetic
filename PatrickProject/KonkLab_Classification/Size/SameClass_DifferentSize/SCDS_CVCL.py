#!/usr/bin/env python3
import os
import sys
import argparse
import random
import math
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict
from scipy.stats import binom_test
from statsmodels.stats.proportion import proportion_confint

# ─── make the top-level repo a Python package root ───
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/Size/SameClass_DifferentSize
REPO_ROOT = os.path.abspath(
    os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, REPO_ROOT)

# ─── hard-coded dataset paths ───
CSV_PATH = os.path.join(REPO_ROOT, 'PatrickProject', 'testdata.csv')
IMG_DIR   = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')

# ─── verify imports ───
from src.utils.model_loader       import load_model
from src.models.feature_extractor import FeatureExtractor

class SizeImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        assert all(col in self.df for col in ['Filename','Class','Size']), \
            "CSV must have Filename, Class, and Size columns"
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, fn, size = row['Class'], row['Filename'], row['Size']
        path = os.path.join(self.img_dir, cls, fn)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), cls, size, idx

def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    classes = [b[1] for b in batch]
    sizes   = [b[2] for b in batch]
    idxs    = [b[3] for b in batch]
    return imgs, classes, sizes, idxs

def main():
    parser = argparse.ArgumentParser("4-way size-prototype eval (same class-different size)")
    parser.add_argument('--model',           default='cvcl-resnext', help="model name")
    parser.add_argument('--seed',     type=int, default=0,             help="random seed")
    parser.add_argument('--device',         default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="compute device")
    parser.add_argument('--batch_size',   type=int, default=64, help="DataLoader batch size")
    parser.add_argument('--trials_per_size', type=int, default=10, help="4-way trials per size")
    parser.add_argument('--max_images',    type=int, default=None,
                        help="if set, subsample this many images")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── load CSV & model ───
    df = pd.read_csv(CSV_PATH)
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)

    # ─── build DataLoader & extract embeddings ───
    ds = SizeImageDataset(CSV_PATH, IMG_DIR, transform)
    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn)

    all_embs, all_classes, all_sizes, all_idxs = [], [], [], []
    with torch.no_grad():
        for imgs, classes, sizes, idxs in loader:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats).cpu()
            all_embs.append(feats)
            all_classes.extend(classes)
            all_sizes.extend(sizes)
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)

    # ─── group indices by (class, size) ───
    cs_idxs = defaultdict(lambda: defaultdict(list))
    for idx, cls, sz in zip(all_idxs, all_classes, all_sizes):
        cs_idxs[cls][sz].append(idx)

    total_correct = 0
    total_trials  = 0

    print("[ℹ️] Running 4-way trials: pick one sample of a given size among same-class, different-size distractors …")
    for cls, size_groups in cs_idxs.items():
        for size, idx_list in size_groups.items():
            # distractors: images of same class but different size
            other_idxs = [
                i for sz2, lst2 in size_groups.items() if sz2 != size
                for i in lst2
            ]
            if len(idx_list) < 1 or len(other_idxs) < 3:
                continue

            correct = 0
            for _ in range(args.trials_per_size):
                q = random.choice(idx_list)
                same_size = [i for i in idx_list if i != q]
                if same_size:
                    proto = all_embs[[all_idxs.index(i) for i in same_size]].mean(0)
                else:
                    proto = all_embs[all_idxs.index(q)]
                proto = proto / proto.norm()

                distractors = random.sample(other_idxs, 3)
                candidates  = [q] + distractors
                feats_cand  = all_embs[[all_idxs.index(i) for i in candidates]]
                sims = feats_cand @ proto
                guess = candidates[sims.argmax().item()]

                correct       += int(guess == q)
                total_correct += int(guess == q)
                total_trials  += 1

            print(f"{cls:12s} / {size:12s}: {correct}/{args.trials_per_size} ({correct/args.trials_per_size:.1%})")

    # ─── summary and statistical test ───
    overall = total_correct / total_trials if total_trials else 0.0
    print(f"\nOverall accuracy: {total_correct}/{total_trials} ({overall:.1%})")

    # 1) Exact binomial test vs. chance p=0.25
    p_val = binom_test(total_correct, total_trials, p=1/4, alternative='greater')
    print(f"P-value (binomial test vs. 25%): {p_val:.4g}")

    # 2) 95% confidence interval (Wilson method)
    ci_low, ci_high = proportion_confint(total_correct, total_trials, alpha=0.05, method='wilson')
    print(f"95% CI for accuracy (Wilson): [{ci_low:.3f}, {ci_high:.3f}]")

    # 3) Approximate normal‐theory CI (for comparison)
    z = 1.96
    se = math.sqrt(overall * (1 - overall) / total_trials)
    norm_low, norm_high = overall - z * se, overall + z * se
    print(f"Approx-normal 95% CI: [{norm_low:.3f}, {norm_high:.3f}]")

if __name__ == '__main__':
    main()

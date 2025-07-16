#!/usr/bin/env python3
import os
import sys
import argparse
import random
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict

# ─── make the top-level repo a Python package root ───
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/SameClass_DifferentColors
REPO_ROOT = os.path.abspath(
    os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, REPO_ROOT)

# ─── dataset paths ───
CSV_PATH = os.path.join(REPO_ROOT, 'PatrickProject', 'testdata.csv')
IMG_DIR  = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')

# ─── verify imports ───
from src.utils.model_loader       import load_model
from src.models.feature_extractor import FeatureExtractor

class ColorImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        assert all(col in self.df for col in ['Filename','Class','Color']), \
            "CSV must have Filename, Class, and Color columns"
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, fn = row['Class'], row['Filename']
        path = os.path.join(self.img_dir, cls, fn)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), row['Class'], row['Color'], idx

def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    classes = [b[1] for b in batch]
    colors  = [b[2] for b in batch]
    idxs    = [b[3] for b in batch]
    return imgs, classes, colors, idxs

def main():
    parser = argparse.ArgumentParser("4-way color-prototype eval")
    parser.add_argument('--model',    default='clip-resnext', help="model name")
    parser.add_argument('--seed',     type=int, default=0,        help="random seed")
    parser.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="compute device")
    parser.add_argument('--batch_size',       type=int, default=64, help="DataLoader batch size")
    parser.add_argument('--trials_per_class', type=int, default=10, help="4-way trials per class")
    parser.add_argument('--max_images',       type=int, default=None,
                        help="if set, subsample this many images")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── load CSV & model ───
    df = pd.read_csv(CSV_PATH)
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)

    # ─── build the full DataLoader ───
    full_ds = ColorImageDataset(CSV_PATH, IMG_DIR, transform)
    full_loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    print(f"[ℹ️] Full dataset size: {len(full_ds)} samples")

    # ─── extract embeddings for all images ───
    all_embs, all_classes, all_colors, all_idxs = [], [], [], []
    with torch.no_grad():
        for imgs, classes, colors, idxs in full_loader:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats)
            feats = feats.float()           # <-- ensure float dtype
            all_embs.append(feats.cpu())
            all_classes.extend(classes)
            all_colors.extend(colors)
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)  # [N, D]

    # ─── run within-class 4-way trials ───
    class_color_idxs = defaultdict(lambda: defaultdict(list))
    for idx, cls, col in zip(all_idxs, all_classes, all_colors):
        class_color_idxs[cls][col].append(idx)

    total_correct = 0
    total_trials  = 0

    print("[ℹ️] Running 4-way color-vs-other-color trials *within each class* …")
    for cls, color_groups in class_color_idxs.items():
        for color, idx_list in color_groups.items():
            # pick distractors only from *other colors* of *the same class*
            other_idxs = [
                i for col2, lst2 in color_groups.items() if col2 != color
                for i in lst2
            ]
            if len(idx_list) < 1 or len(other_idxs) < 3:
                continue

            correct = 0
            for _ in range(args.trials_per_class):
                q = random.choice(idx_list)
                same_color = [i for i in idx_list if i != q]
                proto = all_embs[[all_idxs.index(i) for i in same_color]].mean(0)
                proto = proto / proto.norm()

                distractors = random.sample(other_idxs, 3)
                candidates  = [q] + distractors
                feats_cand  = all_embs[[all_idxs.index(i) for i in candidates]]
                sims = feats_cand @ proto
                guess = candidates[sims.argmax().item()]

                total_correct += int(guess == q)
                total_trials  += 1
                correct       += int(guess == q)

            print(f"{cls:12s} / {color:12s}: {correct}/{args.trials_per_class} "
                  f"({correct/args.trials_per_class:.1%})")

    overall = total_correct / total_trials if total_trials else 0.0
    print(f"\nOverall accuracy: {total_correct}/{total_trials} ({overall:.1%})")

if __name__ == "__main__":
    main()

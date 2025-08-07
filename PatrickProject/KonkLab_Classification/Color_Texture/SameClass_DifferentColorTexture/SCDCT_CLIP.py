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
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/SameClass_DifferentColorTexture
REPO_ROOT = os.path.abspath(
    os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, REPO_ROOT)

# ─── hard‐coded dataset paths ───
CSV_PATH = os.path.join(REPO_ROOT, 'PatrickProject', 'testdata.csv')
IMG_DIR   = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')

# ─── verify imports ───
from src.utils.model_loader       import load_model
from src.models.feature_extractor import FeatureExtractor

class CTImageDataset(Dataset):
    """Loads images + metadata for (class, color, texture)."""
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        required = ['Filename','Class','Color','Texture']
        assert all(c in self.df for c in required), \
            f"CSV must contain columns: {required}"
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, col, tex = row['Class'], row['Color'], row['Texture']
        fn = row['Filename']
        path = os.path.join(self.img_dir, cls, fn)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), cls, col, tex, idx

def collate_fn(batch):
    imgs     = torch.stack([b[0] for b in batch])
    classes  = [b[1] for b in batch]
    colors   = [b[2] for b in batch]
    textures = [b[3] for b in batch]
    idxs     = [b[4] for b in batch]
    return imgs, classes, colors, textures, idxs


def check_existing_results(model_name, test_name):
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV)
        existing = df[(df['Model'] == model_name) & (df['Test'] == test_name)]
        return len(existing) > 0
    return False

def main():
    parser = argparse.ArgumentParser(
        "4-way CLIP eval: same-class, diff-color & texture"
    )
    parser.add_argument('--model',    default='clip-resnext', help="model name")
    parser.add_argument('--seed',     type=int, default=0,        help="random seed")
    parser.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="compute device")
    parser.add_argument('--batch_size',       type=int, default=64, help="DataLoader batch size")
    parser.add_argument('--trials_per_tuple', type=int, default=10, help="trials per (C,Cc,T)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── load CSV + CLIP model ───
    df = pd.read_csv(CSV_PATH)
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)

    # ─── build DataLoader & extract embeddings ───
    ds     = CTImageDataset(CSV_PATH, IMG_DIR, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)
    all_embs, all_meta, all_idxs = [], [], []
    with torch.no_grad():
        for imgs, classes, colors, textures, idxs in loader:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats).cpu()
            feats = feats.float()
            all_embs.append(feats)
            all_meta.extend(zip(classes, colors, textures))
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)

    # ─── group by (class, color, texture) ───
    tuple2idxs = defaultdict(list)
    for idx, meta in zip(all_idxs, all_meta):
        tuple2idxs[meta].append(idx)

    total_correct = 0
    total_trials  = 0
    print("[ℹ️] Running 4-way trials: same class + diff-color & texture distractors")
    for (cls, col, tex), idx_list in tuple2idxs.items():
        # distractors must have same class, but color≠ and texture≠
        distractor_pool = [
            i for m,i in zip(all_meta, all_idxs)
            if m[0] == cls and m[1] != col and m[2] != tex
        ]
        if len(idx_list) < 1 or len(distractor_pool) < 3:
            continue

        correct = 0
        for _ in range(args.trials_per_tuple):
            q = random.choice(idx_list)
            same_rest = [i for i in idx_list if i != q]
            if same_rest:
                proto = all_embs[[all_idxs.index(i) for i in same_rest]].mean(0)
            else:
                proto = all_embs[all_idxs.index(q)]
            proto = proto / proto.norm()

            distractors = random.sample(distractor_pool, 3)
            candidates  = [q] + distractors
            feats_cand  = all_embs[[all_idxs.index(i) for i in candidates]]
            sims = feats_cand @ proto
            guess = candidates[sims.argmax().item()]

            correct       += int(guess == q)
            total_correct += int(guess == q)
            total_trials  += 1

        print(f"{cls:12s} / {col:10s} / {tex:10s} : "
              f"{correct}/{args.trials_per_tuple} "
              f"({correct/args.trials_per_tuple:.1%})")

    overall = total_correct / total_trials if total_trials else 0.0
    print(f"\n[✅] Overall accuracy: {total_correct}/{total_trials} ({overall:.1%})")

if __name__ == '__main__':
    main()

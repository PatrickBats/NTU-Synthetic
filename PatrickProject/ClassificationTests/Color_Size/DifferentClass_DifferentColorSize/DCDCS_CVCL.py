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
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/SameColor_SameSize
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

# ─── hard-coded dataset paths ───
CSV_PATH = os.path.join(REPO_ROOT, 'PatrickProject', 'testdata.csv')
IMG_DIR   = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')

# ─── imports ───
from src.utils.model_loader       import load_model
from src.models.feature_extractor import FeatureExtractor

class CSImageDataset(Dataset):
    """Dataset returning (img_tensor, class, color, size, idx)."""
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        assert all(col in self.df for col in ['Filename','Class','Color','Size']), \
            "CSV must have Filename, Class, Color and Size columns"
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, fn, col, sz = row['Class'], row['Filename'], row['Color'], row['Size']
        path = os.path.join(self.img_dir, cls, fn)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), cls, col, sz, idx

def collate_fn(batch):
    imgs   = torch.stack([b[0] for b in batch])
    classes= [b[1] for b in batch]
    colors = [b[2] for b in batch]
    sizes  = [b[3] for b in batch]
    idxs   = [b[4] for b in batch]
    return imgs, classes, colors, sizes, idxs

def main():
    p = argparse.ArgumentParser("4-way same-color+size, different-class eval")
    p.add_argument('--model',    default='cvcl-resnext', help="model name")
    p.add_argument('--seed',     type=int, default=0,        help="random seed")
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu',
                   help="compute device")
    p.add_argument('--batch_size',       type=int, default=64, help="DataLoader batch size")
    p.add_argument('--trials_per_pair',  type=int, default=10, help="trials per (class,color,size)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) load CSV & model
    df = pd.read_csv(CSV_PATH)
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)
    print(f"[ℹ️] Loaded model '{args.model}'")

    # 2) prepare DataLoader & extract embeddings
    ds = CSImageDataset(CSV_PATH, IMG_DIR, transform)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, collate_fn=collate_fn)

    all_embs, all_cls, all_col, all_sz, all_idxs = [], [], [], [], []
    with torch.no_grad():
        for imgs, classes, colors, sizes, idxs in loader:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats).cpu()
            all_embs.append(feats)
            all_cls .extend(classes)
            all_col .extend(colors)
            all_sz  .extend(sizes)
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)  # [N, D]
    print(f"[ℹ️] Extracted embeddings for {len(all_idxs)} images")

    # 3) group indices by (color,size) → then by class
    cs_map = defaultdict(lambda: defaultdict(list))
    for idx, cls, col, sz in zip(all_idxs, all_cls, all_col, all_sz):
        cs_map[(col,sz)][cls].append(idx)

    total_corr = 0
    total_tr   = 0

    print("[ℹ️] Running same-color+size, different-class 4-way trials…")
    for (col, sz), class_groups in cs_map.items():
        # need at least 2 classes in this bucket
        if len(class_groups) < 2:
            continue
        for cls, idx_list in class_groups.items():
            # need at least one query + ≥3 distractors
            pool = [i for c2, lst in class_groups.items() if c2 != cls for i in lst]
            if len(idx_list) < 1 or len(pool) < 3:
                continue

            correct = 0
            for _ in range(args.trials_per_pair):
                # pick a query
                q = random.choice(idx_list)
                # prototype = mean of other same-cls emb
                others = [i for i in idx_list if i != q]
                proto = (all_embs[[all_idxs.index(i) for i in others]].mean(0)
                         if others else all_embs[all_idxs.index(q)])
                proto = proto / proto.norm()

                # sample 3 from pool (diff class, same col+sz)
                distractors = random.sample(pool, 3)
                cands = [q] + distractors
                feats = all_embs[[all_idxs.index(i) for i in cands]]
                sims  = feats @ proto
                guess = cands[sims.argmax().item()]

                correct += int(guess == q)
                total_corr += int(guess == q)
                total_tr   += 1

            print(f"{col:>8s}/{sz:>6s} • {cls:12s}: {correct}/{args.trials_per_pair}"
                  f" ({correct/args.trials_per_pair:.1%})")

    overall = total_corr / total_tr if total_tr else 0.0
    print(f"\n[✅] Overall accuracy: {total_corr}/{total_tr} ({overall:.1%})")


if __name__ == "__main__":
    main()

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
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/SameClass_DifferentSizeTexture
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

# ─── hard-coded dataset paths ───
CSV_PATH = os.path.join(REPO_ROOT, 'PatrickProject', 'testdata.csv')
IMG_DIR   = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')

# ─── verify imports ───
from src.utils.model_loader       import load_model
from src.models.feature_extractor import FeatureExtractor

class SizeTextureDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        assert all(col in self.df for col in ['Filename','Class','Size','Texture']), \
            "CSV must have Filename, Class, Size, and Texture columns"
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, fn, sz, tex = row['Class'], row['Filename'], row['Size'], row['Texture']
        path = os.path.join(self.img_dir, cls, fn)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), cls, sz, tex, idx

def collate_fn(batch):
    imgs     = torch.stack([b[0] for b in batch])
    classes  = [b[1] for b in batch]
    sizes    = [b[2] for b in batch]
    textures = [b[3] for b in batch]
    idxs     = [b[4] for b in batch]
    return imgs, classes, sizes, textures, idxs

def main():
    p = argparse.ArgumentParser("4-way size+texture prototype eval (same class)")
    p.add_argument('--model',    default='clip-resnext', help="model name (e.g. clip-resnext)")
    p.add_argument('--seed',     type=int, default=0,        help="random seed")
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu',
                   help="compute device")
    p.add_argument('--batch_size',       type=int, default=64, help="DataLoader batch size")
    p.add_argument('--trials_per_group', type=int, default=10,
                   help="4-way trials per (class,size,texture) group")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── load CSV & CLIP model ───
    df = pd.read_csv(CSV_PATH)
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)

    # ─── build DataLoader & extract all embeddings ───
    ds     = SizeTextureDataset(CSV_PATH, IMG_DIR, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)
    all_embs, all_classes, all_sizes, all_textures, all_idxs = [], [], [], [], []
    with torch.no_grad():
        for imgs, classes, sizes, textures, idxs in loader:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats)
            feats = feats.cpu().float()
            all_embs.append(feats)
            all_classes.extend(classes)
            all_sizes.extend(sizes)
            all_textures.extend(textures)
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)  # [N, D]

    # ─── group indices by (class, size, texture) ───
    group_idxs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for idx, cls, sz, tex in zip(all_idxs, all_classes, all_sizes, all_textures):
        group_idxs[cls][sz][tex].append(idx)

    total_correct = 0
    total_trials  = 0

    print("[ℹ️] Running 4-way trials: same class, different size & texture distractors")
    for cls, sizes in group_idxs.items():
        for sz, textures in sizes.items():
            for tex, idx_list in textures.items():
                # require at least one query
                # distractors: same class, size != sz AND texture != tex
                pool = []
                for sz2, texs2 in sizes.items():
                    if sz2 == sz: 
                        continue
                    for tex2, lst in texs2.items():
                        if tex2 == tex: 
                            continue
                        pool.extend(lst)

                if len(idx_list) < 1 or len(pool) < 3:
                    continue

                correct = 0
                for _ in range(args.trials_per_group):
                    q = random.choice(idx_list)
                    others_same = [i for i in idx_list if i != q]
                    if others_same:
                        proto = all_embs[[all_idxs.index(i) for i in others_same]].mean(0)
                    else:
                        proto = all_embs[all_idxs.index(q)]
                    proto = proto / proto.norm()

                    distractors = random.sample(pool, 3)
                    candidates  = [q] + distractors
                    feats_cand  = all_embs[[all_idxs.index(i) for i in candidates]]
                    sims = feats_cand @ proto
                    guess = candidates[sims.argmax().item()]

                    correct      += int(guess == q)
                    total_correct += int(guess == q)
                    total_trials  += 1

                print(f"{cls:12s} / {sz:8s} / {tex:12s}: "
                      f"{correct}/{args.trials_per_group} "
                      f"({correct/args.trials_per_group:.1%})")

    overall = total_correct / total_trials if total_trials else 0.0
    print(f"\nOverall accuracy: {total_correct}/{total_trials} ({overall:.1%})")

if __name__ == "__main__":
    main()

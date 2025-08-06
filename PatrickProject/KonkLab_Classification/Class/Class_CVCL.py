#!/usr/bin/env python3
import os
import sys
import argparse
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F

# ─── Path setup ───
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir))

# Add discover-hidden-visual-concepts to path
DISCOVER_ROOT = os.path.join(REPO_ROOT, 'discover-hidden-visual-concepts')
sys.path.insert(0, DISCOVER_ROOT)
sys.path.insert(0, REPO_ROOT)

# Import from discover-hidden-visual-concepts repo
sys.path.append(os.path.join(DISCOVER_ROOT, 'src'))
from utils.model_loader import load_model
from models.feature_extractor import FeatureExtractor
from models.multimodal.multimodal_lit import MultiModalLitModel

# ─── hard-coded paths ───
CSV_PATH   = os.path.join(REPO_ROOT, 'data', 'KonkLab', 'testdata.csv')
IMG_DIR    = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')
MASTER_CSV = os.path.join(
    REPO_ROOT,
    'PatrickProject',
    'Chart_Generation',
    'all_prototype_results.csv'
)

class ClassImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        assert 'Filename' in self.df and 'Class' in self.df, \
            "CSV needs Filename and Class columns"
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, fn = row['Class'], row['Filename']
        path = os.path.join(self.img_dir, cls, fn)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), cls, idx

def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    classes = [b[1] for b in batch]
    idxs    = [b[2] for b in batch]
    return imgs, classes, idxs

def main():
    parser = argparse.ArgumentParser("4-way class-prototype eval (CVCL)")
    parser.add_argument('--model',            default='cvcl-resnext',
                        help="which CVCL model checkpoint to load")
    parser.add_argument('--seed',     type=int, default=0,
                        help="random seed")
    parser.add_argument('--device',          default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size',      type=int, default=64)
    parser.add_argument('--trials_per_class',type=int, default=10)
    parser.add_argument('--max_images',      type=int, default=None,
                        help="if set, subsample this many images for speed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) load model & transform
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)

    # 2) optionally subsample CSV
    df = pd.read_csv(CSV_PATH)
    if args.max_images and len(df) > args.max_images:
        df = df.sample(n=args.max_images, random_state=args.seed).reset_index(drop=True)

    # 3) load data + extract embeddings
    ds = ClassImageDataset(CSV_PATH, IMG_DIR, transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, collate_fn=collate_fn)
    all_embs, all_classes, all_idxs = [], [], []
    with torch.no_grad():
        for imgs, classes, idxs in dl:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats).cpu().float()
            all_embs.append(feats)
            all_classes.extend(classes)
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)

    # 4) build maps for prototype eval
    idx2class  = {i:c for i,c in zip(all_idxs, all_classes)}
    idx2row    = {i:r for r,i in enumerate(all_idxs)}
    class2idxs = {}
    for i,c in idx2class.items():
        class2idxs.setdefault(c, []).append(i)

    # 5) run 4-way trials
    total_correct = 0
    total_trials  = 0
    print("Running 4-way trials - prototype over class...")
    for cls, idxs in class2idxs.items():
        if len(idxs) < 2:
            continue
        correct = 0
        for _ in range(args.trials_per_class):
            # query
            q = random.choice(idxs)
            # prototype over other same-class images
            proto_idxs = [i for i in idxs if i != q]
            proto = all_embs[[idx2row[i] for i in proto_idxs]].mean(0)
            proto = proto / proto.norm()
            # distractors: any image not of this class
            others = [i for i in all_idxs if idx2class[i] != cls]
            distractors = random.sample(others, 3)
            cands  = [q] + distractors
            sims   = (all_embs[[idx2row[i] for i in cands]] @ proto)
            guess  = cands[sims.argmax().item()]
            if guess == q:
                correct += 1
            total_correct += int(guess == q)
            total_trials  += 1

        acc = correct / args.trials_per_class
        print(f"{cls:20s}: {correct}/{args.trials_per_class} ({acc:.1%})")

    # overall summary
    overall_acc = total_correct / total_trials if total_trials else 0.0
    summary_text = f"\nOverall class-prototype accuracy: {total_correct}/{total_trials} ({overall_acc:.1%})"
    print(summary_text)

    # ─── append to master CSV ───
    row = pd.DataFrame([{
        'Model':    args.model,
        'Test':     'Class-Prototype',
        'Correct':  total_correct,
        'Trials':   total_trials,
        'Accuracy': overall_acc
    }])
    os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
    if os.path.exists(MASTER_CSV):
        row.to_csv(MASTER_CSV, mode='a', header=False, index=False, float_format='%.4f')
    else:
        row.to_csv(MASTER_CSV, index=False, float_format='%.4f')

    print(f"Appended result to {MASTER_CSV}")

if __name__ == "__main__":
    main()

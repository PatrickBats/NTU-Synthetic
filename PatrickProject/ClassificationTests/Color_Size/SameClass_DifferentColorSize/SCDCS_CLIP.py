#!/usr/bin/env python3
import os, sys, argparse, random, torch, pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict

# ─── make the top-level repo a Python package root ───
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/SameClass_DifferentColorSize
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

# ─── hard-coded dataset paths ───
CSV_PATH = os.path.join(REPO_ROOT, 'PatrickProject', 'testdata.csv')
IMG_DIR   = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')

# ─── imports ───
from src.utils.model_loader       import load_model
from src.models.feature_extractor import FeatureExtractor

class ColorSizeDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        assert all(c in self.df for c in ['Filename','Class','Color','Size']), \
            "CSV must have Filename, Class, Color, and Size columns"
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cls, fn, col, sz = row['Class'], row['Filename'], row['Color'], row['Size']
        path = os.path.join(self.img_dir, cls, fn)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), cls, col, sz, i

def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    classes = [b[1] for b in batch]
    colors  = [b[2] for b in batch]
    sizes   = [b[3] for b in batch]
    idxs    = [b[4] for b in batch]
    return imgs, classes, colors, sizes, idxs

def main():
    p = argparse.ArgumentParser("4-way same-class, different-color+size (CLIP)")
    p.add_argument('--model',    default='clip-resnext', help="model name")
    p.add_argument('--seed',     type=int, default=0,        help="random seed")
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu',
                  help="compute device")
    p.add_argument('--batch_size',       type=int, default=64, help="DataLoader batch size")
    p.add_argument('--trials_per_combo', type=int, default=10, help="4-way trials per (color,size)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load model & transform
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)

    # build dataset + dataloader
    ds = ColorSizeDataset(CSV_PATH, IMG_DIR, transform)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4,
                        collate_fn=collate_fn)

    # extract all embeddings
    all_embs, all_classes, all_colors, all_sizes, all_idxs = [], [], [], [], []
    with torch.no_grad():
        for imgs, classes, colors, sizes, idxs in loader:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats).cpu()
            feats = feats.float()
            all_embs.append(feats)
            all_classes.extend(classes)
            all_colors.extend(colors)
            all_sizes.extend(sizes)
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)

    # group by class → (color,size) → indices
    combo_idxs = defaultdict(lambda: defaultdict(list))
    class_idxs = defaultdict(list)
    for idx, cls, col, sz in zip(all_idxs, all_classes, all_colors, all_sizes):
        combo_idxs[cls][(col,sz)].append(idx)
        class_idxs[cls].append(idx)

    total_correct = 0
    total_trials  = 0
    print("[ℹ️] Running same-class, different-color+size trials (CLIP)…")

    for cls, combos in combo_idxs.items():
        for (col,sz), idx_list in combos.items():
            # distractors: same class but BOTH color≠col AND size≠sz
            pool = [i for i in class_idxs[cls]
                    if all_colors[all_idxs.index(i)]  != col
                    and all_sizes[all_idxs.index(i)]   != sz]
            if len(idx_list) < 1 or len(pool) < 3:
                continue

            correct = 0
            for _ in range(args.trials_per_combo):
                q = random.choice(idx_list)
                # build prototype from _other_ same-(col,sz) examples
                peers = [i for i in idx_list if i != q]
                if peers:
                    proto = all_embs[[all_idxs.index(i) for i in peers]].mean(0)
                else:
                    proto = all_embs[all_idxs.index(q)]
                proto = proto / proto.norm()

                # sample 3 distractors
                distractors = random.sample(pool, 3)
                candidates  = [q] + distractors
                feats_cand  = all_embs[[all_idxs.index(i) for i in candidates]]
                sims = feats_cand @ proto
                guess = candidates[sims.argmax().item()]

                correct += int(guess == q)
                total_correct += int(guess == q)
                total_trials  += 1

            print(f"{cls:12s} / {col:12s}·{sz:8s}: "
                  f"{correct}/{args.trials_per_combo} ({correct/args.trials_per_combo:.1%})")

    overall = total_correct/total_trials if total_trials else 0.0
    print(f"\n✅ Overall accuracy: {total_correct}/{total_trials} ({overall:.1%})")


if __name__ == "__main__":
    main()

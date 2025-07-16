#!/usr/bin/env python3
import os, sys, argparse, random, torch, pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict

# ─── make the top-level repo a Python package root ───
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/SameClass_DifferentTexture
REPO_ROOT = os.path.abspath(
    os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, REPO_ROOT)

# ─── hard-coded dataset paths ───
CSV_PATH = os.path.join(REPO_ROOT, 'PatrickProject', 'testdata.csv')
IMG_DIR  = os.path.join(REPO_ROOT, 'data', 'KonkLab', '17-objects')

# ─── verify imports ───
from src.utils.model_loader      import load_model
from src.models.feature_extractor import FeatureExtractor

class TextureImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        assert all(col in self.df for col in ['Filename','Class','Texture']), \
            "CSV must have Filename, Class, and Texture columns"
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, fn, tex = row['Class'], row['Filename'], row['Texture']
        path = os.path.join(self.img_dir, cls, fn)
        img = Image.open(path).convert('RGB')
        return self.transform(img), cls, tex, idx


def collate_fn(batch):
    imgs     = torch.stack([b[0] for b in batch])
    classes  = [b[1] for b in batch]
    textures = [b[2] for b in batch]
    idxs     = [b[3] for b in batch]
    return imgs, classes, textures, idxs


def main():
    parser = argparse.ArgumentParser(
        '4-way texture-prototype eval (same class, diff texture)'
    )
    parser.add_argument('--model',               default='cvcl-resnext',
                        help='model name')
    parser.add_argument('--seed',     type=int,  default=0,
                        help='random seed')
    parser.add_argument('--device',  default='cuda'
                        if torch.cuda.is_available() else 'cpu',
                        help='compute device')
    parser.add_argument('--batch_size',          type=int, default=64,
                        help='DataLoader batch size')
    parser.add_argument('--trials_per_texture',  type=int, default=10,
                        help='4-way trials per texture')
    parser.add_argument('--max_images',          type=int, default=None,
                        help='if set, subsample this many images')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── load dataset & model ───
    df = pd.read_csv(CSV_PATH)
    model, transform = load_model(args.model, seed=args.seed, device=args.device)
    extractor = FeatureExtractor(args.model, model, args.device)

    # ─── build DataLoader & extract embeddings ───
    ds     = TextureImageDataset(CSV_PATH, IMG_DIR, transform)
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )
    all_embs, all_classes, all_textures, all_idxs = [], [], [], []
    with torch.no_grad():
        for imgs, classes, textures, idxs in loader:
            feats = extractor.get_img_feature(imgs.to(args.device))
            feats = extractor.norm_features(feats).cpu()
            all_embs.append(feats)
            all_classes.extend(classes)
            all_textures.extend(textures)
            all_idxs.extend(idxs)
    all_embs = torch.cat(all_embs, dim=0)

    # ─── group indices by (class,texture) ───
    ct_idxs = defaultdict(lambda: defaultdict(list))
    for idx, cls, tex in zip(all_idxs, all_classes, all_textures):
        ct_idxs[cls][tex].append(idx)

    total_correct = 0
    total_trials  = 0

    print('[ℹ️] Running 4-way trials: same class, different texture distractors')
    for cls, tex_groups in ct_idxs.items():
        for tex, idx_list in tex_groups.items():
            # distractors: images with class == cls AND texture != tex
            distractor_pool = [i for i, c, t in zip(all_idxs, all_classes, all_textures)
                               if c == cls and t != tex]
            if len(idx_list) < 1 or len(distractor_pool) < 3:
                continue

            correct = 0
            for _ in range(args.trials_per_texture):
                q = random.choice(idx_list)
                same_tex = [i for i in idx_list if i != q]
                if same_tex:
                    proto = all_embs[[all_idxs.index(i) for i in same_tex]].mean(0)
                else:
                    proto = all_embs[all_idxs.index(q)]
                proto = proto / proto.norm()

                distractors = random.sample(distractor_pool, 3)
                candidates  = [q] + distractors
                feats_cand  = all_embs[[all_idxs.index(i) for i in candidates]]
                sims = feats_cand @ proto
                guess = candidates[sims.argmax().item()]

                total_correct += int(guess == q)
                correct       += int(guess == q)
                total_trials  += 1

            print(f"{cls:12s} / {tex:12s}: {correct}/{args.trials_per_texture} ({correct/args.trials_per_texture:.1%})")

    overall = total_correct / total_trials if total_trials else 0.0
    print(f"\nOverall accuracy: {total_correct}/{total_trials} ({overall:.1%})")

if __name__ == '__main__':
    main()

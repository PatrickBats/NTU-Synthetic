#!/usr/bin/env python3
import os
from pathlib import Path

# ─── CHANGE THIS to your actual path ───────────────────────────────────────────
DATASET_DIR = Path(
    "/home/patrick/ssd/discover-hidden-visual-concepts/"
    "PatrickProject/ImageEditing/third_party/OmniGen2/synthetic_dataset"
)
# ────────────────────────────────────────────────────────────────────────────────

def count_color_dirs(base_path: Path):
    for child in base_path.iterdir():
        if child.is_dir() and child.name.endswith("_color"):
            # count only regular files in that folder
            count = sum(1 for f in child.iterdir() if f.is_file())
            print(f"{child.name}: {count} files")

if __name__ == "__main__":
    count_color_dirs(DATASET_DIR)

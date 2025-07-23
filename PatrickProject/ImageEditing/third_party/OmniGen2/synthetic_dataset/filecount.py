#!/usr/bin/env python3
import os
from pathlib import Path

def count_color_dirs(base_path):
    """
    Walks through base_path and counts files in each directory
    whose name ends with '_color', printing only the folder name.
    """
    base = Path(base_path)
    if not base.is_dir():
        raise ValueError(f"{base_path!r} is not a directory")

    for dirpath, _dirnames, filenames in os.walk(base):
        dir = Path(dirpath)
        if dir.name.endswith("_color"):
            # count only regular files (skip subdirs, symlinks if desired)
            file_count = sum(1 for f in filenames if (dir / f).is_file())
            print(f"{dir.name}: {file_count} files")


if __name__ == "__main__":
    DATASET_PATH = (
        "/home/patrick/ssd/discover-hidden-visual-concepts/"
        "PatrickProject/ImageEditing/third_party/OmniGen2/synthetic_dataset"
    )
    count_color_dirs(DATASET_PATH)

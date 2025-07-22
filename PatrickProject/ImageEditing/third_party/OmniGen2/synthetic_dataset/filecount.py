#!/usr/bin/env python3
import os
from pathlib import Path

def count_color_dirs(base_path):
    """
    Walks through base_path and counts files in each directory
    whose name ends with '_color'.
    """
    base = Path(base_path)
    if not base.is_dir():
        raise ValueError(f"{base_path!r} is not a directory")

    for dirpath, dirnames, filenames in os.walk(base):
        dir_name = Path(dirpath).name
        if dir_name.endswith('_color'):
            # count only regular files (skip subdirs, symlinks if desired)
            file_count = sum(1 for f in filenames if (Path(dirpath) / f).is_file())
            print(f"{dirpath}: {file_count} files")

if __name__ == "__main__":
    DATASET_PATH = "/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/ImageEditing/third_party/OmniGen2/synthetic_dataset"
    count_color_dirs(DATASET_PATH)

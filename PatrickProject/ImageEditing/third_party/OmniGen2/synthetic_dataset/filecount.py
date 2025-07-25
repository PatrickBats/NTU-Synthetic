#!/usr/bin/env python3
import os
import csv
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = Path(
    "/home/patrick/ssd/discover-hidden-visual-concepts/"
    "PatrickProject/ImageEditing/third_party/OmniGen2/synthetic_dataset"
)
CSV_PATH = Path("/home/patrick/ssd/discover-hidden-visual-concepts/unique_class_names.csv")
THRESHOLD = 116
# ────────────────────────────────────────────────────────────────────────────────

def count_color_images(base_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dirpath, _, filenames in os.walk(base_path):
        d = Path(dirpath)
        if d.name.endswith("_color"):
            cls = d.name[:-6]  # strip "_color"
            counts[cls] = sum(1 for f in filenames if (d / f).is_file())
    return counts

def update_csv_in_place(csv_path: Path, counts: dict[str, int]):
    # 1) Read everything first
    with csv_path.open(newline="") as inf:
        reader = csv.DictReader(inf, skipinitialspace=True)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # 2) Rewrite the same file
    with csv_path.open("w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cls = row.get("Class", "")
            row["Colorsmade"] = "yes" if counts.get(cls, 0) > THRESHOLD else ""
            writer.writerow(row)

if __name__ == "__main__":
    counts = count_color_images(DATASET_PATH)
    update_csv_in_place(CSV_PATH, counts)
    print(f"✅ Updated {CSV_PATH} in place.")

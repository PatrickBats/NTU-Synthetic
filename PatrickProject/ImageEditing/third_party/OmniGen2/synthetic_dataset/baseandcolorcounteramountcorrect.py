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
    """
    Walks through base_path and returns a dict mapping
    class_name -> number of files in `<class>_color` folder.
    """
    counts: dict[str, int] = {}
    for dirpath, _, filenames in os.walk(base_path):
        d = Path(dirpath)
        if d.name.endswith("_color"):
            cls = d.name[:-6]  # remove the "_color" suffix
            counts[cls] = sum(1 for f in filenames if (d / f).is_file())
    return counts

def update_csv_in_place(csv_path: Path, color_counts: dict[str, int]):
    """
    Reads all rows, then overwrites the same CSV:
      - Colorsmade = "yes" if color_counts[class] > THRESHOLD
      - Basemade  = "yes" if `<class>_bases` exists under DATASET_PATH
    """
    # 1) Read entire CSV
    with csv_path.open(newline="") as inf:
        reader = csv.DictReader(inf, skipinitialspace=True)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # 2) Rewrite CSV in place
    with csv_path.open("w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cls = row.get("Class", "").strip()
            # Colorsmade logic
            row["Colorsmade"] = "yes" if color_counts.get(cls, 0) > THRESHOLD else ""
            # Basemade logic: check for `<class>_bases` folder
            base_folder = DATASET_PATH / f"{cls}_bases"
            row["Basemade"] = "yes" if base_folder.is_dir() else ""
            writer.writerow(row)

if __name__ == "__main__":
    color_counts = count_color_images(DATASET_PATH)
    update_csv_in_place(CSV_PATH, color_counts)
    print(f"✅ Updated `{CSV_PATH.name}` in place (Colorsmade & Basemade).")

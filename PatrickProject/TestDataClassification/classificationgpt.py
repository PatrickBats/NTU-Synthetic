#!/usr/bin/env python3
import os
import csv
import colorsys
import numpy as np
from PIL import Image
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────
ROOT_DIR     = "/home/patrick/ssd/discover-hidden-visual-concepts/data/KonkLab/17-objects"
FOLDERS_FILE = "/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/folders_to_process.txt"
OUTPUT_CSV   = "/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/color_size_amount_output.csv"

SIZE_THRESH   = {"small": 0.30, "large": 0.70}
MIN_COMP_FRAC = 0.01

# ─── HELPERS ────────────────────────────────────────────────────────────

def hue_to_name(deg: float) -> str:
    if deg < 15 or deg >= 345: return "red"
    if deg < 45:   return "orange"
    if deg < 75:   return "yellow"
    if deg <165:   return "green"
    if deg <195:   return "cyan"
    if deg <255:   return "blue"
    if deg <285:   return "purple"
    if deg <330:   return "pink"
    return "brown"

def compute_color(img: Image.Image) -> str:
    arr  = np.array(img)
    mask = np.any(arr < 245, axis=2)
    px   = arr[mask]
    hsv  = np.array([colorsys.rgb_to_hsv(*(p/255)) for p in px])
    h, s, v = hsv[:,0], hsv[:,1], hsv[:,2]
    avg_s, avg_v = s.mean(), v.mean()

    if avg_v > 0.9 and avg_s < 0.1:
        return "white"
    if avg_v < 0.1:
        return "black"
    if avg_s < 0.1:
        return "gray"

    hist, edges = np.histogram(h, bins=12, range=(0,1))
    idx = hist.argmax()
    if hist[idx] / len(h) >= 0.7:
        center = ((edges[idx] + edges[idx+1]) / 2) * 360
        return hue_to_name(center)
    return "multicolored"

def compute_size(img: Image.Image) -> str:
    arr  = np.array(img)
    mask_frac = np.any(arr < 245, axis=2).sum() / (img.width * img.height)
    if mask_frac < SIZE_THRESH["small"]:
        return "Small"
    if mask_frac > SIZE_THRESH["large"]:
        return "Large"
    return "Medium"

def compute_amount(img: Image.Image) -> str:
    rgb  = np.array(img)
    # build binary mask of “object” pixels
    mask = (np.any(rgb < 245, axis=2)).astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(mask)
    h, w = mask.shape
    min_size = h * w * MIN_COMP_FRAC

    count = 0
    for lab in range(1, num_labels):
        if np.sum(labels == lab) >= min_size:
            count += 1
            if count > 1:
                return "Multiple"
    return "Single"

# ─── I/O ────────────────────────────────────────────────────────────────

def build_image_list():
    items = []
    with open(FOLDERS_FILE) as f:
        cats = [line.strip() for line in f if line.strip()]
    for cat in cats:
        folder = os.path.join(ROOT_DIR, cat)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".jpg", ".png")):
                items.append((cat, fn, os.path.join(folder, fn)))
    return items

def main():
    images = build_image_list()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class","Filename","Color","Size","Amount"])
        for cat, fn, path in images:
            img = Image.open(path).convert("RGB")
            color  = compute_color(img)
            size   = compute_size(img)
            amount = compute_amount(img)
            writer.writerow([cat, fn, color, size, amount])
            print(f"{cat}/{fn} → color={color}, size={size}, amount={amount}")

    print(f"\n✅ Done! CSV written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

import os
import shutil

# 1) Set these paths
source_root = '/home/patrick/ssd/discover-hidden-visual-concepts/data/KonkLab/17-objects'
target_dir  = '/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/fullfolder'

# 2) Create the target folder if it doesn’t exist
os.makedirs(target_dir, exist_ok=True)

# 3) Define which file-types you consider “images”
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

# 4) Walk and copy
for dirpath, _, filenames in os.walk(source_root):
    for fname in filenames:
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            src = os.path.join(dirpath, fname)
            dst = os.path.join(target_dir, fname)

            # avoid overwriting by renaming if needed
            if os.path.exists(dst):
                base, ext = os.path.splitext(fname)
                i = 1
                while True:
                    new_name = f"{base}_{i}{ext}"
                    dst = os.path.join(target_dir, new_name)
                    if not os.path.exists(dst):
                        break
                    i += 1

            shutil.copy2(src, dst)   # <-- copy instead of move

print("All images have been copied to:", target_dir)

import os
import csv

# 1. Point this at your “17-objects” folder
root_dir = '/home/patrick/ssd/discover-hidden-visual-concepts/data/KonkLab/17-objects'

# 2. Collect only the directory names
folder_names = [
    name for name in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, name))
]

# 3. Write them out to CSV
out_csv = 'folders.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['folder_name'])        # header
    for name in folder_names:
        writer.writerow([name])

print(f"Wrote {len(folder_names)} folder names to {out_csv}")

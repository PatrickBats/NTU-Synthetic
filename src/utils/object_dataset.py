from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path
import json
import os
import re


def get_class_names(data_root_dir):
    subfolders = [name for name in os.listdir(data_root_dir)
                if os.path.isdir(os.path.join(data_root_dir, name))]
    return subfolders # as list of class names
    
class KonkTrialDataset(Dataset):
    def __init__(self, trials_file_path, transform=None):
        with open(trials_file_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.class_names = list(set([trial["target_category"] for trial in self.data]))

    def __getitem__(self, idx):
        trial = self.data[idx]
        imgs = []

        # Load and transform the target and foil images
        for filename in [trial["target_img_filename"]] + trial["foil_img_filenames"]:
            # ["target.jpg", "foil1.jpg", "foil2.jpg", "foil3.jpg"]
            try:
                with Image.open(filename).convert("RGB") as img:
                    # img = resize_transform(img)
                    if self.transform:
                        img = self.transform(img)
                    imgs.append(img)
            except IOError:
                print(f"Error: Could not open image {filename}. Skipping.")
                continue

        # Stack images list into a single tensor
        imgs = torch.stack(imgs) #[batch_size, number_of_trial_imgs, channel, height, width]

        label = trial["target_category"] # single label
        foil_labels = trial["foil_categories"] # list of labels
        # because we are doing 4 imgs pick 1 game, and we select highest sim picture based on gt picture text
        return imgs, label, foil_labels

    def __len__(self):
        return len(self.data)


class KonkObjectDataset(Dataset):
    def __init__(self, root_dir, transform=None, split=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # Loading class names
        self.class_names = get_class_names(self.root_dir)
        
        # Load data paths and labels
        self.data = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if split is None:
                for img_path in class_dir.glob('**/*.[jJ][pP][gG]'):
                    self.data.append((img_path, class_name))
            elif split == 'train':
                for img_path in class_dir.glob('*.[jJ][pP][gG]'):
                    if 'TestItems' not in str(img_path):
                        self.data.append((img_path, class_name))
            elif split == 'val':  
                # val set
                test_dir = class_dir / 'TestItems'
                if test_dir.exists():
                    for img_path in test_dir.glob('*.[jJ][pP][gG]'):
                        self.data.append((img_path, class_name))
    
    def __getitem__(self, idx):
        img_path, class_label = self.data[idx]
        img = Image.open(str(img_path)).convert('RGB')  # Load image
        
        if self.transform:
            img = self.transform(img)
            
        label = self.class_names.index(class_label)  # Convert label name to index
        return img, label

    def __len__(self):
        return len(self.data)
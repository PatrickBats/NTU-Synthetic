import os
import torch
import torchvision
from torch.utils.data import Dataset, ConcatDataset 
from .object_dataset import KonkObjectDataset, KonkTrialDataset  

# Dataset root
data_root = os.getenv("DATA_ROOT", default="/home/Dataset/")

DATASET_ROOTS = {
    "imagenet": os.path.join(data_root, "ImageNet/ILSVRC2012"),
    "broden": os.path.join(data_root, "Broden/broden1_224/images"),
    "broden_net-dissect": os.path.join(data_root, "Broden/broden1_224"),
    "objects": os.path.join(data_root, "KonkLab/17-objects"),
}

def get_dataset(dataset_name: str, **kwargs) -> Dataset:
    """Get dataset instance by name.
    
    For ImageNet:
    - split='train': ~1.2M training images
    - split='val': ~50K validation images (commonly used as test set)
    - split=None: combines both splits
    
    Note: The original ImageNet test set (~100K images) remains private 
    and is not available in public distributions.
    """
    # print(f"Dataset: {dataset_name}")
    
    if dataset_name == 'objects':
        return KonkObjectDataset(root_dir=DATASET_ROOTS['objects'], transform=kwargs['transform'])
    
    elif dataset_name == 'object-trial':
        return KonkTrialDataset(trials_file_path=kwargs['trials_file_path'], transform=kwargs['transform'])
    
    
    elif dataset_name.split('_')[0] == 'imagenet':
        # First try to get split from kwargs, if not found, try to get from dataset_name
        split = kwargs.get('split') or (
            dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else None
        )
        if split:
            if split not in ['train', 'val']:
                raise ValueError("ImageNet only supports 'train' or 'val' splits. "
                               "The original test set remains private.")
            return torchvision.datasets.ImageNet(
                root=DATASET_ROOTS['imagenet'],
                split=split,
                **{k: v for k, v in kwargs.items() if k != 'split'}
            )
        else:
            # For whole dataset, combine train and val
            return ConcatDataset([
                torchvision.datasets.ImageNet(root=DATASET_ROOTS['imagenet'], transform=kwargs.get('transform'), split='train'),
                torchvision.datasets.ImageNet(root=DATASET_ROOTS['imagenet'], transform=kwargs.get('transform'), split='val')
            ])
    
    elif dataset_name == 'cifar100':
        return torchvision.datasets.CIFAR100(
            root=os.path.expanduser("~/.cache"),
            train=kwargs.get('split', 'train') == 'train',
            transform=kwargs.get('transform')
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def trial_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])  # Stack the images
    labels = [item[1] for item in batch]  # Keep target labels as a list
    foil_labels = [item[2] for item in batch]  # Keep foil labels as a list of lists

    return imgs, labels, foil_labels
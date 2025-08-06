import os
import json
import math
import random
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from . import dataset_loader 
from . import similarity
from . import model_loader

import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import re

def get_object_classes(data_root_dir, vocab_path="data/multimodal/vocab.json", match_type='full'):
    """
    Get all class names from the data directory and categorize them into seen and unseen classes
    based on a vocabulary file.
    
    Parameters:
    -----------
    data_root_dir : str
        Path to the directory containing class folders
    vocab_path : str, optional
        Path to the vocabulary JSON file, default is "data/multimodal/vocab.json"
    match_type : str, optional
        Type of matching to perform ('full' or 'partial'), default is 'full'
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'all': list of all class names
        - 'seen': list of class names present in the vocabulary
        - 'unseen': list of class names not present in the vocabulary
    """
    # Get all class names from directory
    subfolders = [name for name in os.listdir(data_root_dir)
                 if os.path.isdir(os.path.join(data_root_dir, name))]
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = set(json.load(f).keys())
    
    # Categorize classes based on match type
    if match_type == 'full':
        seen_classes = [c for c in subfolders if c in vocab]
        unseen_classes = [c for c in subfolders if c not in vocab]
    elif match_type == 'partial':
        seen_classes = [c for c in subfolders if set(re.compile(r'\W+').split(c)) & vocab]
        unseen_classes = [c for c in subfolders if not set(re.compile(r'\W+').split(c)) & vocab]
    else:
        raise ValueError(f"Unknown match_type: {match_type}. Use 'full' or 'partial'.")
    
    return {
        'all': subfolders,
        'seen': seen_classes,
        'unseen': unseen_classes
    }   

def read_trial_json(trial_path):
    with open(trial_path, 'r') as f:
        data = json.load(f)
    return data

def read_image(img_path, transform=None):
    img = Image.open(img_path)
    if transform is not None:
        img = transform(img)
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        if isinstance(img, np.ndarray) and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
    return img

def show_trial_img(trial, transform=None):

    target_img = read_image(trial["target_img_filename"], transform=transform)
    foil_imgs = [read_image(foil, transform=transform) for foil in trial["foil_img_filenames"]]
    foil_classes = trial["foil_categories"]
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    axs[0].imshow(target_img)
    axs[0].set_title(f"Target: {trial['target_category']}")
    axs[0].axis('off')

    for i, (foil_img, foil_class) in enumerate(zip(foil_imgs, foil_classes)):
        axs[i+1].imshow(foil_img)
        axs[i+1].set_title(f"Foil {i+1}: {foil_class}")
        axs[i+1].axis('off')

    plt.tight_layout()
    return plt.show()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def get_trial_accuracy(predictions):
    correct_pred = 0
    total_pred = 0
    correct_cls_pred = defaultdict(int)
    total_cls_predic = defaultdict(int)

    # Iterate over all collected predictions
    for pred in predictions:
        label = pred['gt_label']
        is_correct = pred['is_correct']

        # Increment total predictions count
        total_pred += 1
        total_cls_predic[label] += 1

        # Check if the prediction was correct
        if is_correct:
            correct_pred += 1
            correct_cls_pred[label] += 1

    # Calculate overall accuracy
    acc = correct_pred / total_pred if total_pred > 0 else 0

    # Calculate per-class accuracy
    cls_acc = {cls: (correct_cls_pred[cls] / total_cls_predic[cls] if total_cls_predic[cls] > 0 else 0)
               for cls in total_cls_predic}

    return acc, cls_acc

def get_results_dir(args_dict, results_root_dir='../experiments/trial/'):
    trial_type = args_dict.get('trial_type', 'na')
    model = args_dict.get('model', 'na')   
    use_concept = args_dict.get('use_concept', False)

    concept_prefix = 'concept_' if use_concept else ''
    folder_name = f"{concept_prefix}{model}_{trial_type}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    results_dir_path = os.path.join(results_root_dir, folder_name)
    os.makedirs(results_dir_path, exist_ok=True)  

    return results_dir_path

def save_trial_results(results_dir, args_dict, predictions, csv_save_path='overall_acc.csv'):
    """Save trial results to specified directory"""
    # Check and create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Check and create parent directory for csv_save_path if it doesn't exist
    csv_dir = os.path.dirname(csv_save_path)
    if csv_dir:  # if csv_save_path includes a directory path
        os.makedirs(csv_dir, exist_ok=True)

    overall_acc, class_acc = get_trial_accuracy(predictions)

    # result board
    save_overall_acc(results_dir, args_dict, overall_acc, csv_save_path)

    # Combine overall and per-class accuracies
    combined_acc = {
        "overall_accuracy": overall_acc,
        "class_accuracies": class_acc
    }

    # save overall and perclass acc to json
    with open(os.path.join(results_dir, 'accuracies.json'), 'w') as file:
        json.dump(combined_acc, file, indent=4)

    # save args
    with open(os.path.join(results_dir, 'args.json'), 'w') as file:
        json.dump(args_dict, file, indent=4)

    # save per-trial predictions to csv
    predictions_df = pd.DataFrame(predictions)
    with open(os.path.join(results_dir, 'predictions.csv'), 'w') as file:
        predictions_df.to_csv(file, index=True) 
    
    print(f"Results saved to {results_dir}")

def save_overall_acc(results_dir, args_dict, overall_acc, csv_save_path):
    data = args_dict.copy()
    data['overall_accuracy'] = overall_acc
    data['results_dir'] = results_dir

    file_path = os.path.join(csv_save_path)

    # Define the desired column order
    desired_order = ['overall_accuracy', 'model'] + [col for col in data if col not in ['overall_accuracy', 'model']]

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        
        # Ensure all columns from data are in existing_df
        for col in data:
            if col not in existing_df.columns:
                existing_df[col] = None
        
        # Check for duplication
        duplicate = existing_df.apply(lambda row: all(row[col] == data.get(col, None) for col in existing_df.columns), axis=1)
        if duplicate.any():
            print("Duplication found. Skipping addition of this row.")
            return

        # Create a new row with the current data, matching existing columns
        new_row = pd.DataFrame({col: [data.get(col, None)] for col in existing_df.columns})
        
        # Concatenate the existing DataFrame with the new row
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        # If file doesn't exist, create a new DataFrame with all columns from data
        updated_df = pd.DataFrame([data])

    # Reorder columns
    all_columns = list(updated_df.columns)
    ordered_columns = [col for col in desired_order if col in all_columns] + [col for col in all_columns if col not in desired_order]
    updated_df = updated_df[ordered_columns]

    # Save the updated DataFrame
    updated_df.to_csv(file_path, index=False)

    print(f"Overall Accuracy updated in {file_path}")


"""
Adapt from CLIP-Dissect
Source: https://github.com/Trustworthy-ML-Lab/CLIP-dissect/blob/main/utils.py
"""

PM_SUFFIX = {"max":"_max", "avg":""}

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                             PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name


# take target_name as one more input to determine which forward function should be used
def save_target_activations(target_model, dataset, save_name, target_layers, batch_size, device, pool_mode='avg'):
    _make_save_dir(os.path.dirname(save_name))
    save_names = {layer: save_name.format(layer) for layer in target_layers}
    
    if _all_saved(save_names):
        print("All specified layers' activations are already saved.")
        return

    # Dictionary to hold feature vectors for each target layer
    all_features = {layer: [] for layer in target_layers}

    hooks = {}
    # Register hooks to capture activations
    successful_layers = []
    failed_layers = []
    for layer in target_layers:
        module = dict(target_model.named_modules()).get(layer)
        if module:
            hooks[layer] = module.register_forward_hook(get_activation(all_features[layer], pool_mode))
            successful_layers.append(layer)
        else:
            failed_layers.append(layer)
    
    if successful_layers:
        print(f"Successfully registered hooks for layers: {', '.join(successful_layers)}")
    if failed_layers:
        print(f"Warning: These layers do not exist in the model: {', '.join(failed_layers)}")

    # Forward pass through the model to capture activations
    with torch.no_grad():
        for images, _ in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            images = images.to(device)
            if hasattr(target_model, 'encode_image'):
                _ = target_model.encode_image(images.to(device))
            else:
                _ = target_model(images.to(device)) 
    
    # Save captured features and remove hooks
    for layer in target_layers:
        torch.save(torch.cat(all_features[layer]), save_names[layer])
        hooks[layer].remove()

    # Free memory
    del all_features
    torch.cuda.empty_cache()
    print("Activations saved and memory cleaned up.")
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    target_model, target_preprocess = model_loader.load_model(model_name=target_name, device=device)
    # setup data
    data_c = dataset_loader.get_dataset(d_probe, transform=clip_preprocess)
    data_t = dataset_loader.get_dataset(d_probe, transform=target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    # ignore empty lines
    words = [i for i in words if i!=""]
    
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_names = get_save_names(clip_name = clip_name, target_name = target_name,
                                target_layer = '{}', d_probe = d_probe, concept_set = concept_set,
                                pool_mode=pool_mode, save_dir = save_dir)
    target_save_name, clip_save_name, text_save_name = save_names
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode)
    return
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True, device="cuda"):
    
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name, map_location='cpu')
    similarity = similarity_fn(clip_feats, target_feats, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


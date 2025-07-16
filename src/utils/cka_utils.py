import os
import torch
from src.utils.utils import get_save_names, save_target_activations
from src.utils.dataset_loader import get_dataset
from datetime import datetime

def save_cka_activations(model_name, model, transform, layers, d_probe, batch_size, device, pool_mode, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = get_dataset(d_probe, transform=transform) # val for ImageNet val

    save_name, _, _ = get_save_names(clip_name = 'NA', target_name = model_name,
                                target_layer = '{}', d_probe = d_probe, concept_set = 'NA',
                                pool_mode=pool_mode, save_dir = save_dir)
    
    # print(f"Target Model: {model.__class__.__name__}\n Data Len:{len(data)} \n Target Save Name: {save_name} \n Target Layers: {layers} \n Pool Mode: {pool_mode} \n Batch Size: {batch_size} \n Device: {device}") 

    save_target_activations(model, data, save_name, layers,
                            batch_size, device, pool_mode) 
    return

def save_cka_matrix(d_probe, activation_dir, model_layers_1, model_layers_2, args):
    cka_matrix = compute_CKA_layer_matrix(
        data_name=d_probe, 
        activation_dir=activation_dir, 
        model_name_y=args.model_y, 
        model_name_x=args.model_x, 
        model_y_layers=model_layers_1,
        model_x_layers=model_layers_2)
    
    result_dir = args.result_dir

    # Save the CKA matrix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")
    
    # Create a dictionary with matrix and metadata
    cka_data = {
        'matrix': cka_matrix,
        'model_y': args.model_y,  # Y-axis model
        'model_x': args.model_x,  # X-axis model
        'model_y_layers': model_layers_1,  # Y-axis layer names
        'model_x_layers': model_layers_2,  # X-axis layer names
        'dataset': d_probe
    }
    
    save_path = os.path.join(
        result_dir, 
        f"mat_{args.model_y}_{args.model_x}_{d_probe}_{timestamp}.pt"
    )
    torch.save(cka_data, save_path)
    print(f"CKA matrix and model information saved to: {save_path}")


def get_layer_prefix(model):
    model_name = model.__class__.__name__
    if model_name == 'MultiModalLitModel':
        return 'vision_encoder.model.'
    elif model_name == 'CLIP':
        return 'visual.'
    else:
        return ''
    
def get_cka_layers(model, layers):
    """Get in depth layer names for CKA computation
    - model: loaded model
    - layers: list of layer names(1st level)
    """
    prefix = get_layer_prefix(model)
    cka_layers = []
    for layer in layers:
        cka_layer = prefix + layer
        cka_layers.append(cka_layer)
    return cka_layers

def compute_CKA_similarity(X, Y):
    # Center the activation matrices
    X_centered = X - torch.mean(X, dim=0, keepdim=True)
    Y_centered = Y - torch.mean(Y, dim=0, keepdim=True)

    # Compute Gram matrices
    K = X_centered @ X_centered.T
    L = Y_centered @ Y_centered.T

    # Compute HSIC
    HSIC = torch.sum(K * L)

    # Compute normalization terms
    HSIC_X = torch.sum(K * K)
    HSIC_Y = torch.sum(L * L)

    # Compute CKA similarity
    CKA_similarity = HSIC / torch.sqrt(HSIC_X * HSIC_Y)

    return CKA_similarity

def compute_CKA_layer_matrix(data_name, activation_dir, model_name_y, model_name_x, model_y_layers, model_x_layers):
    """Compute CKA similarity matrix between two models' layers
    Args:
        data_name: Name of dataset used
        activation_dir: Directory containing saved activations
        model_name_y: Name of first model, y axis of matrix
        model_name_x: Name of second model, x axis of matrix
        model_y_layers: List of layer names from first model
        model_x_layers: List of layer names from second model
    """
    num_layers_1 = len(model_y_layers)
    num_layers_2 = len(model_x_layers)
    cka_matrix = torch.zeros((num_layers_1, num_layers_2))
    
    for i, layer1 in enumerate(model_y_layers):
        # Load activations from model 1
        activation_1 = os.path.join(activation_dir, f"{data_name}_{model_name_y}_{layer1}.pt")
        feat_1 = torch.load(activation_1, map_location='cpu').float()

        for j, layer2 in enumerate(model_x_layers):
            # Load activations from model 2
            activation_2 = os.path.join(activation_dir, f"{data_name}_{model_name_x}_{layer2}.pt")
            feat_2 = torch.load(activation_2, map_location='cpu').float()

            # Compute CKA similarity between layers
            cka_similarity = compute_CKA_similarity(feat_1, feat_2)
            cka_matrix[i, j] = cka_similarity

    return cka_matrix

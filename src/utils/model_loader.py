import torch.nn as nn
import clip
from torch.nn import init
from torchvision.models import resnet50, resnext50_32x4d, ResNeXt50_32X4D_Weights
from ..models.multimodal.multimodal_lit import MultiModalLitModel
from huggingface_hub import hf_hub_download
import os
import torch
import sys

def show_available_models():
    available_models = ['cvcl-resnext', 'cvcl-random',
                       'resnext', 'resnext-random',
                       'clip','clip-res',
                       'dino_s_resnext50', 
                       ]
    return available_models

def replace_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, nn.GELU())
        else:
            replace_gelu(child)

def randomize_vision_encoder_weights(model):
    for name, param in model.named_parameters():
        if 'vision_encoder' in name:
            if 'bn' in name:
                if 'weight' in name:
                    init.constant_(param, 1.0)  # set weight to 1
                elif 'bias' in name:
                    init.constant_(param, 0.0)  # set bias to 0
            else:
                if param.dim() > 1:
                    init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')  # conv layers use He initialization
                else:
                    init.zeros_(param)  # set bias to 0
                    
def randomize_resnet_weights(model): # also applicable to resnext
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
       
def load_model(model_name, seed=0, device='cuda'):
    """returns target model and its transform function, only for evaluation"""
    
    model_parts = model_name.split("-")
    
    if model_parts[0] == "cvcl":  
        if "random" in model_name:
            model, transform = MultiModalLitModel.load_model('cvcl-resnext')
            randomize_vision_encoder_weights(model)
        else: # resnext and vit
            model, transform = MultiModalLitModel.load_model(model_name, seed, device)

        if "vit" in model_name:
            replace_gelu(model)
            
        model.to(device)
    
    elif model_parts[0].startswith("dino") :
        # ['dino_say_resnext50', 'dino_s_resnext50', 'dino_a_resnext50', 'dino_y_resnext50']
        arch, patch_size = "resnext50_32x4d", None
        checkpoint = hf_hub_download(repo_id="eminorhan/"+model_parts[0], filename=model_parts[0]+".pth")
        model = build_dino_mugs(arch, patch_size)
        load_dino_mugs(model, checkpoint, "teacher")
        
        model.to(device)
        from torchvision import transforms as pth_transforms
        transform = pth_transforms.Compose([
            pth_transforms.Resize(224), # TODO: check img size with DINO
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        if "random" in model_name:
            randomize_resnet_weights(model)
            print(f"Successfully randomize weights for {model_name.split('-')[0]}")
        
    elif model_parts[0] == "clip":
        if "res" in model_name:
            backbone = "RN50" 
        else:
            backbone = "ViT-L/14" # source: CVCL Supplementary Materials
        model, transform = clip.load(f"{backbone}", device=device)
        
    elif model_parts[0] == "resnext":
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT).to(device)
        transform = ResNeXt50_32X4D_Weights.IMAGENET1K_V2.transforms()
        # model.fc = nn.Linear(model.fc.in_features,4)
        model.to(device)
        
        if "random" in model_name:
            randomize_resnet_weights(model)
            print(f"Successfully randomize weights for {model_name}")
    
    elif model_name=="resnet":
        model_name_cap = model_name.replace("resnet", "ResNet")
        weights = eval("models.{}50_Weights.IMAGENET1K_V1".format(model_name_cap)) # ResNet50 by default
        transform = weights.transforms()
        model = eval("models.{}(weights=weights).to(device)".format(model_name))
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.eval()
    
    return model, transform


def get_default_full_layers(model_name):
    if model_name == 'cvcl-resnext' or model_name == 'cvcl-random':
        return ['vision_encoder.model.layer1', 'vision_encoder.model.layer2', 'vision_encoder.model.layer3', 'vision_encoder.model.layer4']
    elif model_name == 'clip-res':
        return ['visual.layer1', 'visual.layer2', 'visual.layer3', 'visual.layer4']
    elif model_name == 'resnext':
        return ['layer1', 'layer2', 'layer3', 'layer4']
    elif model_name.startswith("dino"):
        return ['layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise ValueError(f"No predefined layer from model: {model_name}")
    
def load_dino_mugs(model, pretrained_weights, checkpoint_key):
    """Source: https://github.com/eminorhan/silicon-menagerie/blob/master/utils.py"""
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `encoder.` prefix if it exists
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")
        
def build_dino_mugs(arch, patch_size):
    """Source: https://github.com/eminorhan/silicon-menagerie/blob/master/utils.py"""
    import models.vision_transformer_dino_mugs as vits
    from torchvision import models as torchvision_models

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    if arch in vits.__dict__.keys():
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    # otherwise, we check if the architecture is in torchvision models
    elif arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[arch]()
        model.fc = torch.nn.Identity()
    else:
        print(f"Unknown architecture: {arch}")
        sys.exit(1)

    return model
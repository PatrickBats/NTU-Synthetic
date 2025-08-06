import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import settings
import torch
import torchvision
from huggingface_hub import hf_hub_download
from src.models.multimodal.multimodal_lit import MultiModalLitModel
import clip
from torchvision.models import resnet50, resnext50_32x4d, ResNeXt50_32X4D_Weights


def loadmodel(hook_fn):
    # Load CVCL
    if settings.MODEL == 'cvcl':
        checkpoint_name = "cvcl_s_dino_resnext50_embedding"
        checkpoint = hf_hub_download(repo_id="wkvong/"+checkpoint_name, filename=checkpoint_name+".ckpt")
        model = MultiModalLitModel.load_from_checkpoint(checkpoint_path=checkpoint)
    elif settings.MODEL == 'clip':
        model, _ = clip.load("RN50", device="cuda")
    elif settings.MODEL == 'resnext':
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT).to(device="cuda")
    else:
        # original model load
        if settings.MODEL_FILE is None:
            model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
        else:
            checkpoint = torch.load(settings.MODEL_FILE)
            if isinstance(checkpoint, (dict, torch.OrderedDict)):
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
                if settings.MODEL_PARALLEL:
                    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
    
    for name in settings.FEATURE_NAMES:
        module = dict(model.named_modules()).get(name)
        if module:
            module.register_forward_hook(hook_fn)
            print(f"Hook registered for layer: {name}")
        else:
            print(f"Warning: Layer {name} does not exist in the model.")
    if settings.GPU:
        model.cuda()
    
    model.eval()
    return model
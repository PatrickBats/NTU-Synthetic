
"""
Adapt from CLIP-Dissect
Source: https://github.com/Trustworthy-ML-Lab/CLIP-dissect/blob/main/utils.py
"""

def get_activation(outputs, mode, trial_size=None):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    trial_size: number of imgs inside single trial
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                if trial_size is not None: # for trials
                    output = output.view(-1, trial_size, output.size(-3), output.size(-2), output.size(-1))
                    outputs.append(output.mean(dim=[3, 4]).detach())
                else: 
                    outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                if trial_size is not None: # for trials
                    output = output.view(-1, trial_size, output.size(-3), output.size(-2), output.size(-1))
                    outputs.append(output.amax(dim=[3, 4]).detach())
                else:
                    outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='raw':
        def hook(model, input, output):
            outputs.append(output.detach())
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected one of 'avg', 'max', 'raw'.")
    return hook

def register_hooks(model, layers, mode='avg', trial_size=None):
    activations = {layer: [] for layer in layers} #TODO not start with layer
    hooks = {}

    # Register forward hook
    for layer in layers:
        module = dict(model.named_modules()).get(layer)
        if module:
            hooks[layer] = module.register_forward_hook(get_activation(activations[layer], mode, trial_size))
            # print(f"Hook registered for layer: {layer}")
        else:
            print(f"Warning: Layer '{layer}' does not exist in the model.")

    return activations, hooks

def remove_hooks(hooks):
    for layer in hooks:
        hooks[layer].remove()
        # print(f"Hook removed for layer: {layer}")

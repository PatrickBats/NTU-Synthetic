import os
import argparse
import datetime
import json
import pandas as pd
import torch

import src.utils.utils as utils
import src.utils.similarity as similarity 

"""adapt from CLIP-Dissect"""

parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default="cvcl-resnext", 
                    choices=["cvcl-resnext","cvcl-random","resnext","clip-res","dino_say_resnext50","dino_s_resnext50","dino_a_resnext50","dino_y_resnext50"],
                   help=""""Which model to dissect.""")
parser.add_argument("--target_layers", type=str, default="vision_encoder.model.layer1,vision_encoder.model.layer2,vision_encoder.model.layer3,vision_encoder.model.layer4",
                    # default: conv1,layer1,layer2,layer3,layer4
                    # cvcl: vision_encoder.model.layer1,vision_encoder.model.layer2,vision_encoder.model.layer3,vision_encoder.model.layer4
                    # clip: visual.layer1,visual.layer2,visual.layer3,visual.layer4
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--d_probe", type=str, default="broden", 
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden", "objects"])
parser.add_argument("--concept_set", type=str, default="data/baby+30k+konk.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="experiments/saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="experiments/neuron_concepts", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder", 
                                                                               "cos_similarity", "cos_similarity_cubed"])

parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    if "," in args.target_layers:
        args.target_layers = [layer.strip() for layer in args.target_layers.split(",")]    
    else:
        args.target_layers = [args.target_layers]
    
    similarity_fn = eval("similarity.{}".format(args.similarity_fn))
    
    utils.save_activations(clip_name = args.clip_model, target_name = args.target_model, 
                           target_layers = args.target_layers, d_probe = args.d_probe, 
                           concept_set = args.concept_set, batch_size = args.batch_size, 
                           device = args.device, pool_mode=args.pool_mode, 
                           save_dir = args.activation_dir)
    
    outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}
    with open(args.concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    
    # compare similarity for each layer
    for target_layer in args.target_layers: #! per-layer matching
        save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                  target_layer = target_layer, d_probe = args.d_probe,
                                  concept_set = args.concept_set, pool_mode = args.pool_mode,
                                  save_dir = args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names

        similarities = utils.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=False, device=args.device
        )
        vals, ids = torch.max(similarities, dim=1)
        
        del similarities
        torch.cuda.empty_cache()
        
        descriptions = [words[int(idx)] for idx in ids]
        
        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer]*len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())
        
    df = pd.DataFrame(outputs)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    
    model_prefix_map = {
        "clip-res": "clip_res",
        "cvcl-resnext": "cvcl",
        "cvcl-random": "cvcl-random",
        "resnext": "resnext",
        "dino_s_resnext50": "dino_s", 
    }
    
    model_prefix = model_prefix_map.get(args.target_model, args.target_model)
    
    # Get concept set name from path
    concept_name = os.path.splitext(os.path.basename(args.concept_set))[0]
    
    # Create standardized naming format
    save_name = f"{model_prefix}_{args.d_probe}_{concept_name}"
    save_path = os.path.join(args.result_dir, save_name)
    
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "descriptions.csv"), index=False)
    with open(os.path.join(save_path, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(f"Results saved to {save_path}")
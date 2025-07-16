import argparse
from torch.utils.data import DataLoader

from src.utils.generate_trials import get_trials
from src.utils.dataset_loader import get_dataset, trial_collate_fn
from src.utils.model_loader import load_model, get_default_full_layers, show_available_models
from src.utils.utils import set_seed, save_trial_results, get_results_dir
from src.utils.predictors import TrialPredictor
from src.models.feature_extractor import FeatureExtractor


def main(args):
    set_seed(args.seed)
    args_dict = vars(args)

    ### STEP 1: Get Trials (either generate or load)
    trial_path = get_trials(args.trial_type, args.seed, args.num_img_per_trial, args.num_trials_per_image, args.class_type, args.map_file, args.object_resize)
    args_dict.setdefault('trial_path', trial_path) # add trial path in args

    ### STEP 2: Load model, and load dataset from json trial file
    model_name = args.model
    model, transform = load_model(model_name, args.seed, args.device) # seed for different checkpoint 
    print(f"Model loaded: {model_name} on device: {args.device}")
    
    data = get_dataset(dataset_name='object-trial', trials_file_path=trial_path, transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=trial_collate_fn)
    
    ### STEP 3: Initialize the trial classifier
    feature_extractor = FeatureExtractor(model_name, model, args.device)
    classifier = TrialPredictor(feature_extractor)
    
    ### STEP 4: Predict the trial results
    if args.map_file is not None: # using neurons
        layers = get_default_full_layers(model_name) if args.layers is None else args.layers.split(",")
        args_dict.setdefault('layers', layers)
        predictions = classifier.predict_using_neurons(dataloader, layers, args.map_file, args.top_k)
    else: 
        predictions = classifier.predict(dataloader) 
    
    ### STEP 5: Save the results
    results_dir = get_results_dir(args_dict, args.exps_root_dir)
    save_trial_results(results_dir, args_dict, predictions, args.csv_save_path) #TODO: save selected neurons
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVCL-TrialZeroShot')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')

    parser.add_argument('--model', type=str, default='cvcl-resnext', choices=show_available_models(), help='Model name with different backbones')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    parser.add_argument('--trial_type', type=str, default='custom', choices=['pub','custom'], help='choose published trial or custom generated trial')
    parser.add_argument('--class_type', type=str, default='unseen', choices=['unseen','seen','full'], help='choose unseen, seen or full class type')
    parser.add_argument('--object_resize', action='store_true', help='Resize the object images')
    parser.add_argument('--num_img_per_trial', type=int, default=4, help='Number of images per trial')
    parser.add_argument('--num_trials_per_image', type=int, default=5, help='How many trials generated for same target image')
    
    # no default value for layers, map_file, top_k
    parser.add_argument('--layers', type=str, help='List of layers to consider for concept mapping. Example: "layer1,layer2,layer3,layer4".')
    parser.add_argument('--top_k', type=int, default=1, help='Top k concepts to consider across all layers/Top k neurons to consider')
    parser.add_argument('--map_file', type=str, help='Path to the neuron concept mapping file(dissect csv file)')
    
    parser.add_argument('--csv_save_path', type=str, default='./results/overall_acc.csv', help='Path to save the overall accuracy csv file')
    parser.add_argument('--exps_root_dir', type=str, default='./experiments/trials/', help='Path to save the results')
    
    args = parser.parse_args()
    main(args)
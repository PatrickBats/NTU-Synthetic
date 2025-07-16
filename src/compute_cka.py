from src.utils.model_loader import load_model
import argparse
from utils.cka_utils import get_cka_layers, save_cka_activations, save_cka_matrix

parser = argparse.ArgumentParser(description='CKA similarity compute')

parser.add_argument("--model_y", type=str, default="cvcl-resnext", 
                    choices=['clip-res', 'cvcl-resnext', 'resnext'],
                    help="First(Y axis in matrix) model in the pair for comparison")
parser.add_argument("--model_x", type=str, default="clip-res", 
                    choices=['clip-res', 'cvcl-resnext', 'resnext'],
                    help="Second(X axis in matrix) model in the pair for comparison")
parser.add_argument("--layers", type=str, default="layer1,layer2,layer3,layer4",
                    help="Layer names separated by commas (no spaces)")
parser.add_argument("--d_probe", type=str, default="imagenet_val", 
                    choices = ["objects", "imagenet_val", "broden"])
parser.add_argument("--batch_size", type=int, default=128, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda:0", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="./experiments/CKA/activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="./experiments/CKA/similarity_matrix", help="where to save calculated CKA matrix")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")

parser.parse_args()


if __name__ == '__main__':
    args = parser.parse_args()

    device = args.device

    model_y, transform1 = load_model(model_name=args.model_y, device=device)
    model_x, transform2 = load_model(model_name=args.model_x, device=device)

    layers = args.layers.split(',')
    model_layers_y = get_cka_layers(model_y, layers)
    model_layers_x = get_cka_layers(model_x, layers)  

    d_probe = args.d_probe
    batch_size = args.batch_size
    activation_dir = args.activation_dir
    pool_mode = args.pool_mode

    save_cka_activations(args.model_y, model_y, transform1, model_layers_y, d_probe, batch_size, device, pool_mode, activation_dir)
    save_cka_activations(args.model_x, model_x, transform2, model_layers_x, d_probe, batch_size, device, pool_mode, activation_dir)
    
    save_cka_matrix(d_probe, activation_dir, model_layers_y, model_layers_x, args)
    
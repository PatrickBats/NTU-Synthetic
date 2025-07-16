import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
from util.save_settings import save_settings
import argparse

parser = argparse.ArgumentParser(description='Net-Dissect')

parser.add_argument("--device", type=str, default="cuda:0")

parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()
    
    device = args.device

    fo = FeatureOperator()
    model = loadmodel(hook_feature)

    model.to(device)
    ### Add: save args ###
    save_settings()
    ############ STEP 1: feature extraction ###############
    features, maxfeature = fo.feature_extraction(device=device, model=model)

    for layer_id,layer in enumerate(settings.FEATURE_NAMES):
    ############ STEP 2: calculating threshold ############
        thresholds = fo.quantile_threshold(features[layer_id],savepath=f"quantile_{layer_id}.npy")

    ############ STEP 3: calculating IoU scores ###########
        tally_result = fo.tally(features[layer_id],thresholds,savepath=f"tally_{layer_id}.csv")

    ############ STEP 4: generating results ###############
        generate_html_summary(fo.data, layer,
                            tally_result=tally_result,
                            maxfeature=maxfeature[layer_id],
                            features=features[layer_id],
                            thresholds=thresholds)
        if settings.CLEAN:
            clean()
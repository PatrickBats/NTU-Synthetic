#!/bin/bash

cd "$(dirname "$0")/.."

# Define model configurations
declare -A MODEL_LAYERS=(
    ["cvcl-resnext"]="vision_encoder.model.layer1,vision_encoder.model.layer2,vision_encoder.model.layer3,vision_encoder.model.layer4"
    ["cvcl-random"]="vision_encoder.model.layer1,vision_encoder.model.layer2,vision_encoder.model.layer3,vision_encoder.model.layer4"
    ["clip-res"]="visual.layer1,visual.layer2,visual.layer3,visual.layer4"
    ["resnext"]="layer1,layer2,layer3,layer4"
    ["dino_s_resnext50"]="layer1,layer2,layer3,layer4"
)

# Common parameters
d_probe="objects"
concept_set="data/baby+30k+konk.txt"
device="cuda:0"
batch_size=256

# Save exps
result_dir="./experiments/neuron_labeling/labeled_neurons"
activation_dir="./experiments/neuron_labeling/saved_activations"


# Loop through models
for model in "${!MODEL_LAYERS[@]}"; do
    echo "Processing model: $model"
    python -m src.describe_neurons \
        --similarity_fn soft_wpmi \
        --target_model "$model" \
        --target_layers "${MODEL_LAYERS[$model]}" \
        --d_probe "$d_probe" \
        --concept_set "$concept_set" \
        --device "$device" \
        --result_dir "$result_dir" \
        --activation_dir "$activation_dir"\
        --batch_size "$batch_size"
done


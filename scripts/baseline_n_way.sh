#!/bin/bash

# change directory to the root of the project 
cd "$(dirname "$0")/.."

seeds=(0 1 2) 
models=(
    'cvcl-resnext'
    'clip-res'
) 
device='cuda:0'
num_img_per_trial=(2 4 8 16 32) 
class_type=("seen" "unseen" "full") 
csv_save_path=("./experiments/trials/results/baseline.csv")
exps_root_dir=("./experiments/trials/baseline/")
max_batch_size=256
min_batch_size=5  

for model in "${models[@]}"
do
    # Define which class types to run based on the model
    if [ "$model" = "cvcl-resnext" ]; then
        class_types=("seen") 
    else
        class_types=("${class_type[@]}")  
    fi
    
    for curr_class_type in "${class_types[@]}"
    do
        for num_img in "${num_img_per_trial[@]}"
        do
            batch_size=$(( 4 * max_batch_size / num_img ))
            batch_size=$(( batch_size < min_batch_size ? min_batch_size : batch_size ))
            for seed in "${seeds[@]}" 
            do 
                python -m src.trial \
                    --model "$model" \
                    --seed "$seed" \
                    --device "$device" \
                    --batch_size "$batch_size" \
                    --num_img_per_trial "$num_img" \
                    --class_type "$curr_class_type" \
                    --csv_save_path "$csv_save_path" \
                    --exps_root_dir "$exps_root_dir"      
            done
        done
    done
done
#!/bin/bash
seeds=(0 1 2)
models=(
        "cvcl-resnext" 
        'cvcl-random'
        'clip-res'
        'resnext'
        'dino_s_resnext50'
 )

device="cuda:0"
num_img_per_trial=(2 4 8 16 32) 
class_type=("seen" "unseen" "full") 

# neuron concepts suffix
dataset_name="objects"
concept_set_name="baby+30k+konk"
csv_save_path=("./experiments/trials/results/neuron_classification.csv")
exps_root_dir=("./experiments/trials/neurons/")
max_batch_size=800
min_batch_size=5  

# model and map_file prefix
declare -A model_map_prefixes=(
    ["clip-res"]="clip_res"
    ["cvcl-random"]="cvcl-random"
    ["cvcl-resnext"]="cvcl"
    ["resnext"]="resnext"
    ["dino_s_resnext50"]="dino_s"
)

for class_type in "${class_type[@]}"
do
    for model in "${models[@]}"
    do
        # generate map_file according to the selected model and suffix
        map_file="./experiments/neuron_labeling/labeled_neurons/${model_map_prefixes[$model]}_${dataset_name}_${concept_set_name}/descriptions.csv"

        for num_img in "${num_img_per_trial[@]}"
        do
            # calculate dynamic batch size and round down
            batch_size=$(( 4 * max_batch_size / num_img ))
            # ensure batch size is between min_batch_size and max_batch_size
            batch_size=$(( batch_size < min_batch_size ? min_batch_size : batch_size ))

            for seed in "${seeds[@]}" 
            do 
                python -m src.trial \
                    --model $model \
                    --top_k 1 \
                    --map_file $map_file \
                    --seed $seed \
                    --device $device \
                    --batch_size $batch_size \
                    --num_img_per_trial $num_img \
                    --class_type $class_type \
                    --csv_save_path $csv_save_path \
                    --exps_root_dir "$exps_root_dir" \

            done
        done
    done
done
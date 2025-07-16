#!/bin/bash
seeds=(0) 
models=(
        "cvcl-resnext" 
)

device="cuda:0"
num_img_per_trial=(4) 
class_type=("unseen") 

# neuron concepts suffix
dataset_name="objects"
concept_set_name="baby+30k+konk"
csv_save_path=("./experiments/trials/results/appendix_4_way.csv")
exps_root_dir=("./experiments/trials/neurons/appendix_4_way/")
max_batch_size=200
min_batch_size=5  


declare -A model_map_prefixes=(
    ["cvcl-resnext"]="cvcl"
)

for class_type in "${class_type[@]}"
do
    for model in "${models[@]}"
    do
        map_file="./experiments/neuron_labeling/labeled_neurons/${model_map_prefixes[$model]}_${dataset_name}_${concept_set_name}/descriptions.csv"

        for num_img in "${num_img_per_trial[@]}"
        do
            batch_size=$(( 4 * max_batch_size / num_img ))
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
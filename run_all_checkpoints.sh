#  #!/bin/bash

# # running all tests
# for combination in "gemma-1.1-7b-it gemma-1.1-7b-it-ckpt" "Meta-Llama-3-8B-Instruct Llama-3-8B-ckpt" "Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-ckpt"; do
# set -- $combination
# model_name1=$1
# model_name2=$2
# for fold_size in 500 1000 1500 2000 2500 3000 4000; do
# for i in {1..10}; do
#     if [[ $model_name2 == Llama-3-8B-ckpt && $i -le 4 ]]; then
#         python -u main.py --exp test --no_wandb --model_name1 "$model_name1" --model_name2 "${model_name2}$i" --fold_size $fold_size --seed2 seed2000 > test_${model_name1}_${model_name2}_${i}_${fold_size}_output.txt 2>&1 &
#     else
#         python -u main.py --exp test --no_wandb --model_name1 "$model_name1" --model_name2 "${model_name2}$i" --fold_size $fold_size > test_${model_name1}_${model_name2}_${i}_${fold_size}_output.txt 2>&1 &
#     fi
# done
# wait
# done
# done


# Define the model combinations
declare -a combinations=(
    "Meta-Llama-3-8B-Instruct Llama-3-8B-ckpt"
    "gemma-1.1-7b-it gemma-1.1-7b-it-ckpt"
    "Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-ckpt"
)

# Define noise values
declare -a noise_values=(0.01 0.05)

# Define fold sizes
declare -a fold_sizes=(500 1000 1500 2000 2500 3000 4000)

# Loop through noise values first
for noise in "${noise_values[@]}"; do
    echo "Starting experiments with noise value: $noise"
    
    # Loop through all combinations
    for combination in "${combinations[@]}"; do
        read -r model_name1 model_name2 <<< "$combination"
        
        for fold_size in "${fold_sizes[@]}"; do
            for i in {1..10}; do
                if [[ $model_name2 == "Llama-3-8B-ckpt" && $i -le 4 ]]; then
                    python main.py \
                        exp=test \
                        metric.test_params.noise=$noise \
                        tau1.model_id=$model_name1 \
                        tau2.model_id="${model_name2}${i}" \
                        metric.test_params.fold_size=$fold_size \
                        tau2.gen_seed=seed2000 \
                        logging.use_wandb=false \
                        > "test_${model_name1}_${model_name2}_${i}_${fold_size}_${noise}_output.txt" 2>&1 &
                else
                    python main.py \
                        exp=test \
                        metric.test_params.noise=$noise \
                        tau1.model_id=$model_name1 \
                        tau2.model_id="${model_name2}${i}" \
                        metric.test_params.fold_size=$fold_size \
                        logging.use_wandb=false \
                        > "test_${model_name1}_${model_name2}_${i}_${fold_size}_${noise}_output.txt" 2>&1 &
                fi
            done
            wait
        done
    done
    
    echo "Completed all experiments with noise value: $noise"
done

echo "All experiments completed"



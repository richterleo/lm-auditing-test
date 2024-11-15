#!/bin/bash

# Define the model combinations
declare -a combinations=(
    "Meta-Llama-3-8B-Instruct Llama-3-8B-ckpt"
    "gemma-1.1-7b-it gemma-1.1-7b-it-ckpt"
    "Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-ckpt"
)

# Define noise values
declare -a noise_values=(0)

# Define fold sizes
declare -a fold_sizes=(2000 3000 4000)

# Loop through noise values first
for noise in "${noise_values[@]}"; do
    echo "Starting experiments with noise value: $noise"
    
    # Loop through fold sizes sequentially
    for fold_size in "${fold_sizes[@]}"; do
        echo "Starting experiments with fold size: $fold_size"
        
        # Loop through all combinations (in parallel)
        for combination in "${combinations[@]}"; do
            read -r model_name1 model_name2 <<< "$combination"
            
            # Run all iterations for this combination in parallel
            for i in {1..10}; do
                echo "Launching experiment: Model1=$model_name1, Model2=${model_name2}${i}, Fold=$fold_size, Noise=$noise"
                
                if [[ $model_name2 == "Llama-3-8B-ckpt" && $i -le 4 ]]; then
                    python main.py \
                        experiments=test_toxicity \
                        test_params.noise=$noise \
                        tau1.model_id=$model_name1 \
                        tau2.model_id="${model_name2}${i}" \
                        test_params.fold_size=$fold_size \
                        tau2.gen_seed=seed2000 \
                        logging.use_wandb=false \
                        > "test_${model_name1}_${model_name2}_${i}_${fold_size}_${noise}_output.txt" 2>&1 &
                else
                    python main.py \
                        experiments=test_toxicity \
                        test_params.noise=$noise \
                        tau1.model_id=$model_name1 \
                        tau2.model_id="${model_name2}${i}" \
                        test_params.fold_size=$fold_size \
                        logging.use_wandb=false \
                        > "test_${model_name1}_${model_name2}_${i}_${fold_size}_${noise}_output.txt" 2>&1 &
                fi
            done
        done
        
        # Wait for all parallel processes to complete before moving to next fold size
        wait
        echo "Completed experiments for fold size: $fold_size"
    done
    
    echo "Completed all experiments with noise value: $noise"
done

echo "All experiments completed"
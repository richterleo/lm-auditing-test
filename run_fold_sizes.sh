#!/bin/bash

# Define the fold sizes to test
declare -a fold_sizes=(1000 2000 3000 4000)

# Function to get the appropriate seed for a checkpoint
get_seed() {
    local checkpoint=$1
    if [[ $checkpoint -le 4 ]]; then
        echo "seed2000"
    else
        echo "seed1000"
    fi
}

# Loop through fold sizes sequentially
for fold_size in "${fold_sizes[@]}"; do
    echo "Starting experiments with fold size: $fold_size"

    # Run all checkpoints in parallel
    for checkpoint in {1..10}; do
        seed=$(get_seed $checkpoint)
        echo "Launching experiment for checkpoint $checkpoint with fold size $fold_size using $seed"
        
        python main.py \
            experiments=test_toxicity \
            test_params.fold_size=$fold_size \
            tau1.model_id=Meta-Llama-3-8B-Instruct \
            tau2.model_id="Llama-3-8B-ckpt${checkpoint}" \
            tau2.gen_seed=$seed \
            metric.metric=toxicity \
            logging.use_wandb=false \
            > "test_ckpt${checkpoint}_fold${fold_size}_output.txt" 2>&1 &
    done
    
    # Wait for all parallel processes to complete before moving to next fold size
    wait
    echo "Completed experiments for fold size: $fold_size"
done

echo "All experiments completed" 
#!/bin/bash

# Define the seeds for each checkpoint based on available data
declare -A checkpoint_seeds
checkpoint_seeds[1]="1000 2000 3000"
checkpoint_seeds[2]="1000 2000 3000"
checkpoint_seeds[3]="2000 3000 4000"
checkpoint_seeds[4]="1000 2000 7000"
checkpoint_seeds[5]="1000 5000 6000"
checkpoint_seeds[6]="1000 6000 7000"
checkpoint_seeds[7]="1000 6000"  # Skip checkpoint 7 as it only has one seed
checkpoint_seeds[8]="1000 6000 7000"
checkpoint_seeds[9]="1000 6000 7000"
checkpoint_seeds[10]="1000 5000 6000"

# Function to run a single comparison
run_comparison() {
    local checkpoint=$1
    local seed1=$2
    local seed2=$3
    
    echo "Running comparison for checkpoint ${checkpoint}: seed${seed1} vs seed${seed2}"
    
    python main.py \
        experiments=test_toxicity \
        test_params.fold_size=2000 \
        tau1.model_id="Llama-3-8B-ckpt${checkpoint}" \
        tau2.model_id="Llama-3-8B-ckpt${checkpoint}" \
        tau1.gen_seed="seed${seed1}" \
        tau2.gen_seed="seed${seed2}" \
        logging.use_wandb=false \
        > "seed_comparison_ckpt${checkpoint}_seed${seed1}_vs_seed${seed2}_output.txt" 2>&1
}

# Run one comparison for each checkpoint
for checkpoint in "${!checkpoint_seeds[@]}"; do
    seeds=(${checkpoint_seeds[$checkpoint]})
    num_seeds=${#seeds[@]}
    
    if [ $num_seeds -lt 2 ]; then
        echo "Skipping checkpoint $checkpoint - insufficient seeds for comparison"
        continue
    fi
    
    # Just take the first two seeds for each checkpoint
    seed1=${seeds[0]}
    seed2=${seeds[1]}
    
    run_comparison $checkpoint $seed1 $seed2
done

echo "All seed comparisons completed" 
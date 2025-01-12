#!/bin/bash

# Function to run experiment with specific parameters
run_experiment() {
    local checkpoint=$1
    local seed1=$2
    local seed2=$3
    
    echo "Running experiment for checkpoint $checkpoint with seeds $seed1 and $seed2"
    
    /path/to/your/python main.py \
        experiments=test_toxicity \
        test_params.fold_size=4000 \
        tau1.model_id="Llama-3-8B-ckpt${checkpoint}" \
        tau2.model_id="Llama-3-8B-ckpt${checkpoint}" \
        tau1.gen_seed="seed${seed1}" \
        tau2.gen_seed="seed${seed2}" \
        logging.use_wandb=false \
        > "false_positive_ckpt${checkpoint}_seed${seed1}_seed${seed2}_output.txt" 2>&1 &
}

# Run experiments for checkpoints 1-4 with seed2000 and seed7000
for i in {1..4}; do
    run_experiment $i 2000 7000
done

# Run experiment for checkpoint 5 with seed1000 and seed6000
run_experiment 5 1000 6000

# Run experiments for checkpoints 6-8-9 with seed1000 and seed7000
for i in 6 8 9; do
    run_experiment $i 1000 7000
done

# Run experiment for checkpoint 10 with seed1000 and seed5000
run_experiment 10 1000 5000

# Wait for all experiments to complete
wait

echo "All false positive experiments completed" 
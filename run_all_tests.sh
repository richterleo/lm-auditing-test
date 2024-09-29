 #!/bin/bash

# running all tests
for combination in "gemma-1.1-7b-it gemma-1.1-7b-it-ckpt" "Meta-Llama-3-8B-Instruct Llama-3-8B-ckpt" "Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-ckpt"; do
set -- $combination
model_name1=$1
model_name2=$2
for fold_size in 500 1500; do
for i in {1..10}; do
    if [[ $model_name2 == Llama-3-8B-ckpt && $i -le 4 ]]; then
        python -u main.py --exp test --no_wandb --model_name1 "$model_name1" --model_name2 "${model_name2}$i" --fold_size $fold_size --seed2 seed2000 > test_${model_name1}_${model_name2}_${i}_${fold_size}_output.txt 2>&1 &
    else
        python -u main.py --exp test --no_wandb --model_name1 "$model_name1" --model_name2 "${model_name2}$i" --fold_size $fold_size > test_${model_name1}_${model_name2}_${i}_${fold_size}_output.txt 2>&1 &
    fi
done
wait
done
done



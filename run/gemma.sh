#!/usr/bin/env bash
#SBATCH --nodelist=firefly1,firefly2
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --cpus-per-task=32
#SBATCH --job-name="gemma3"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1

MODELS=(
    google/t5gemma-2-270m-270m
    google/t5gemma-2-1b-1b
    google/t5gemma-2-4b-4b
)

METHODS=(
    fine-tune
    prefix-tune
    lora
)

for MODEL in ${MODELS[@]}; do
    for METHOD in ${METHODS[@]}; do
        uv run cli \
            --log-level DEBUG
            --method $METHOD
            --model $MODEL
            --task seq2seq
    done
done

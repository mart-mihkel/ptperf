#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="gemma3"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

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
        uv run --no-sync cli \
            --log-level DEBUG \
            --method $METHOD \
            --model $MODEL \
            --task seq2seq
    done
done

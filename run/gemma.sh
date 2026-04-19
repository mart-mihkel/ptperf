#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="t5gemma2"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

MODELS=(
    google/t5gemma-2-270m-270m
    google/t5gemma-2-1b-1b
    google/t5gemma-2-4b-4b
)

for MODEL in ${MODELS[@]}; do
    uv run --no-sync cli \
        --method fine-tune \
        --log-level DEBUG \
        --task causal-lm \
        --model $MODEL

    uv run --no-sync cli \
        --num-virtual-tokens 10 \
        --method prefix-tune \
        --log-level DEBUG \
        --task causal-lm \
        --model $MODEL

    uv run --no-sync cli \
        --log-level DEBUG \
        --task causal-lm \
        --max-steps 1024 \
        --model $MODEL
done

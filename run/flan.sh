#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="flan"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

MODELS=(
    google/flan-t5-small
    google/flan-t5-base
    google/flan-t5-large
    google/flan-t5-xl
    google/flan-t5-xxl
)

for MODEL in ${MODELS[@]}; do
    uv run --no-sync cli \
        --method fine-tune \
        --log-level DEBUG \
        --task causal-lm \
        --max-steps 1024 \
        --model $MODEL

    uv run --no-sync cli \
        --num-virtual-tokens 10 \
        --method prefix-tune \
        --log-level DEBUG \
        --task causal-lm \
        --max-steps 1024 \
        --model $MODEL

    uv run --no-sync cli \
        --log-level DEBUG \
        --task causal-lm \
        --max-steps 1024 \
        --method lora \
        --model $MODEL
done

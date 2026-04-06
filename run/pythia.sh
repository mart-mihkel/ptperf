#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="pythia"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

MODELS=(
    EleutherAI/pythia-70m
    EleutherAI/pythia-160m
    EleutherAI/pythia-410m
    EleutherAI/pythia-1b
    EleutherAI/pythia-1.4b
    EleutherAI/pythia-2.8b
    EleutherAI/pythia-6.9b
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
            --task causal-lm
    done
done

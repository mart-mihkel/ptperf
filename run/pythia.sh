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
        uv run cli \
            --log-level DEBUG
            --method $METHOD
            --model $MODEL
            --task causal
    done
done

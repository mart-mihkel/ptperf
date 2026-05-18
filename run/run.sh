#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="gptneox"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

MODELS=(
    EleutherAI/pythia-70m
    EleutherAI/pythia-160m
    EleutherAI/pythia-410m
    EleutherAI/pythia-1b
)

for MODEL in ${MODELS[@]}; do
    uv run --no-sync cli \
        --log-level debug \
        --task causal-lm \
        --method fine-tune \
        --model $MODEL

    uv run --no-sync cli \
        --log-level debug \
        --task causal-lm \
        --method lora \
        --lora-alpha 32 \
        --lora-rank 16 \
        --model $MODEL

    uv run --no-sync cli \
        --log-level debug \
        --task causal-lm \
        --method prefix-tune \
        --virtual-tokens 100 \
        --model $MODEL

    uv run --no-sync cli \
        --log-level debug \
        --task causal-lm \
        --method prompt-tune \
        --virtual-tokens 100 \
        --model $MODEL

    uv run --no-sync cli \
        --log-level debug \
        --task causal-lm \
        --method p-tune \
        --virtual-tokens 32 \
        --encoder-dim 512 \
        --model $MODEL
done

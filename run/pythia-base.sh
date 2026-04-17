MODELS=(
    EleutherAI/pythia-70m
    EleutherAI/pythia-410m
    EleutherAI/pythia-1b
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
        --method lora \
        --model $MODEL
done
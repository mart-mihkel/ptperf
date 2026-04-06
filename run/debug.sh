#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --job-name="debug"
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=00:00:10
#SBATCH --gres=gpu:1
#SBATCH --mem=1GB

uv run --no-sync python - <<'EOF'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())
print("torch compiled with cuda:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())
EOF

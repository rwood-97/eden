#!/bin/bash
#SBATCH --job-name=eden-data-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/rag_distillation_%j.out
#SBATCH --error=logs/rag_distillation_%j.err

set -euo pipefail

source .venv/bin/activate

python -m eden.synth_data_generation.generate_rag_distillation \
    --source-type advice \
    --source-dir data/raw \
    --chroma-dir data/chroma \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --save-path data/sft/ \
    --n-records 4 \
    --pairs-per-record 2

python -m eden.synth_data_generation.generate_rag_distillation \
    --source-type plants \
    --source-dir data/raw \
    --chroma-dir data/chroma \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --save-path data/sft/ \
    --n-records 4
    --pairs-per-record 2

python -m eden.synth_data_generation.generate_rag_distillation \
    --source-type pests \
    --source-dir data/raw \
    --chroma-dir data/chroma \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --save-path data/sft/ \
    --n-records 4 \
    --pairs-per-record 2

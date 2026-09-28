#!/usr/bin/env bash
#SBATCH --job-name=qwen-fs-target
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=e.liao.2@umail.leidenuniv.nl
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

cd "$HOME/audio_llm"

module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source .venv-qwen/bin/activate

mkdir -p logs outputs/fewshot

python run_qwen_fewshot.py \
  target_items.yaml \
  target_audio \
  --output outputs/fewshot/qwen_target_results.jsonl \
  --limit 45

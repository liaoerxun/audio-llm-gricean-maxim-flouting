#!/usr/bin/env bash
#SBATCH --job-name=kimi-fs-target
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=e.liao.2@umail.leidenuniv.nl
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$HOME/audio_llm"

module purge
module load ALICE/default
module load CUDA/12.4.0
module load Python/3.11.5-GCCcore-13.2.0

source .venv-kimi/bin/activate

export TOKENIZERS_PARALLELISM=false

mkdir -p logs outputs/fewshot

python run_kimi_fewshot.py \
  target_items.yaml \
  target_audio \
  --output outputs/fewshot/kimi_target_results.jsonl \
  --limit 45

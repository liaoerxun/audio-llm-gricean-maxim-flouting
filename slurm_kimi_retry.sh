#!/usr/bin/env bash
#SBATCH --job-name=retry-kimi
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$HOME/audio_llm"

module purge
module load ALICE/default
module load CUDA/12.4.0
module load Python/3.11.5-GCCcore-13.2.0

source .venv-kimi/bin/activate
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs logs

python retry_failed_kimi.py \
  pilot_items.yaml \
  pilot_audio \
  --source-results outputs/kimi_results.jsonl \
  --output outputs/kimi_retry_results.jsonl

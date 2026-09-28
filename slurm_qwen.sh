#!/usr/bin/env bash
#SBATCH --job-name=pilot-qwen
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

cd "$HOME/audio_llm"



module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1



source .venv-qwen/bin/activate

mkdir -p outputs

python run_qwen.py pilot_items.yaml pilot_audio \
  --output outputs/qwen_results.jsonl "$@"

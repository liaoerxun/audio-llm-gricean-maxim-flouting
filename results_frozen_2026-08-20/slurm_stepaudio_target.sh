#!/usr/bin/env bash
#SBATCH --job-name=target-step
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
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0



source .venv-step/bin/activate


export TOKENIZERS_PARALLELISM=false
export STEP_AUDIO_REPO="$HOME/audio_llm/Step-Audio2"
export PYTHONPATH="${STEP_AUDIO_REPO}:{PYTHONPATH:-}"

mkdir -p outputs

python run_stepaudio.py \
	target_items.yaml \
	target_audio \
	--model-path "${STEP_AUDIO_REPO}/Step-Audio-2-mini" \
	--output outputs/stepaudio_target_results.jsonl \
	"$@"

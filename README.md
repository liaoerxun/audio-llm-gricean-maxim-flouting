# Audio-LLM implicature pilot pipeline

This package runs the 15 pilot items in three conditions on three open-weight
audio-language models. It creates **45 tasks per model**:

- `transcript_only`: text target, no audio (descriptive baseline)
- `neutral_audio`: neutral recording; expected literal reading
- `marked_audio`: prosodically marked recording; expected implicature and item maxim

The model prompt and decoding settings are held constant across models. Raw
responses, normalized predictions, parsing warnings, errors, and latency are
retained in append-only JSONL files.

## 1. Arrange the input files

Copy the final 15-item YAML to this directory as `pilot_items.yaml`. Each item
must contain `item_id` (or `blind_id`), `context`, `target_utterance`, and
`final_maxim` (or `maxim`). `preceding_utterance` is optional.

Create this layout:

```text
audio_llm_pilot_pipeline/
├── pilot_items.yaml
└── pilot_audio/
    ├── QL01_neutral.wav
    ├── QL01_marked.wav
    ├── QL02_neutral.wav
    ├── QL02_marked.wav
    └── ...
```

Filenames must exactly match the YAML IDs. Export from Audacity as mono,
16-bit PCM WAV, 44.1 or 48 kHz. Do not normalize neutral and marked files
independently if this would erase an experimentally meaningful intensity cue.

## 2. Validate before loading any model

```bash
python -m venv .venv-common
source .venv-common/bin/activate
pip install -r requirements-common.txt

python validate_pilot.py pilot_items.yaml pilot_audio
python dry_run.py pilot_items.yaml pilot_audio
wc -l outputs/pilot_tasks.jsonl
```

Stop unless validation says `passed`, `item_count` is 15, `task_count` is 45,
`audio_file_count` is 30, and the final command prints 45.

Open `outputs/pilot_tasks.jsonl` and spot-check at least one item from each
maxim. Audio conditions deliberately omit the target transcript so that the
audio model must process the recording.

## 3. Install each model in a separate environment

### Qwen2-Audio-7B-Instruct

```bash
python -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -r requirements-qwen.txt
```

### Kimi-Audio-7B-Instruct

Follow the CUDA/PyTorch versions required by the official Kimi-Audio repository,
then:

```bash
git clone https://github.com/MoonshotAI/Kimi-Audio.git
python -m venv .venv-kimi
source .venv-kimi/bin/activate
pip install -e ./Kimi-Audio
pip install PyYAML
```

### Step-Audio-2-mini

Follow the official repository's CUDA installation instructions (its reference
environment uses Python 3.10+, PyTorch 2.3 with CUDA 12.1, and Transformers
4.49.0), then:

```bash
git clone https://github.com/stepfun-ai/Step-Audio2.git
python -m venv .venv-step
source .venv-step/bin/activate
pip install -r ./Step-Audio2/requirements.txt
pip install PyYAML
export STEP_AUDIO_REPO="$PWD/Step-Audio2"
```

On ALICE, add the cluster's required `module load` and `#SBATCH --partition` /
`#SBATCH --account` lines to the three templates. Model weights may need to be
downloaded on a login/data-transfer node before compute jobs start.

## 4. Smoke test: exactly one item × three conditions

Run the three jobs with `--limit 3`:

```bash
sbatch slurm_qwen.sh --limit 3
sbatch slurm_kimi.sh --limit 3
sbatch slurm_stepaudio.sh --limit 3
```

After completion, each result file should contain three successful records:

```bash
wc -l outputs/qwen_results.jsonl \
      outputs/kimi_results.jsonl \
      outputs/stepaudio_results.jsonl
python summarize.py outputs/*_results.jsonl \
  --csv outputs/smoke_results.csv \
  --summary outputs/smoke_summary.json
```

Inspect `status`, `prediction`, `validation_warnings`, and `raw_response`. Fix
installation or parsing problems before proceeding. Do **not** alter the prompt
separately for one model.

## 5. Full pilot

Submit the same scripts without `--limit`:

```bash
sbatch slurm_qwen.sh
sbatch slurm_kimi.sh
sbatch slurm_stepaudio.sh
```

The runners resume automatically: successful task IDs already in the JSONL are
skipped, while errored tasks are retried. Since the three smoke-test records are
retained, each final file should contain 45 successful task IDs, not 48. If a
task previously failed, its error record is deliberately retained, so the raw
file can contain more than 45 lines.

Then combine the outputs:

```bash
python summarize.py \
  outputs/qwen_results.jsonl \
  outputs/kimi_results.jsonl \
  outputs/stepaudio_results.jsonl
```

Expected total: 135 successful task IDs = 15 items × 3 conditions × 3 models.
Earlier retained error records may increase the raw row count.

## 6. Pilot decision rule

First check technical validity: missing outputs, parse warnings, ASR failures,
clipping, or model-specific crashes. Next compare neutral versus marked readings
within each item. Treat transcript-only as a baseline, not as having a forced
gold label. Do not select final stimuli solely because these same three models
performed well; use the pilot to identify technical or stimulus defects and
document every replacement or rerecording.

Keep the validated YAML, validation report, task manifest, exact prompts/code,
environment versions, raw JSONL, summary files, and job logs for reproducibility.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import build_tasks, load_items, run_tasks
from run_qwen import QwenBackend


RETRY_TASK_IDS = {
    "B067::transcript_only",
    "B075::transcript_only",
}

FORMAT_CORRECTION = """
IMPORTANT FORMAT CORRECTION:
Choose exactly one value for every field.
Do not copy the list of available choices.
Do not invent another maxim.
Return exactly one complete JSON object:
{"reading":"literal","maxim":"none","confidence":0.8,"auditory_cue":"none"}

Replace reading with exactly literal or implicature.
Replace maxim with exactly quality, quantity, manner, or none.
For this transcript-only task, auditory_cue must be exactly none.
Return the JSON object only.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry selected malformed Qwen target outputs."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qwen_target_retry_results.jsonl"),
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2-Audio-7B-Instruct",
    )
    parser.add_argument("--extension", default="wav")
    args = parser.parse_args()

    all_tasks = build_tasks(
        load_items(args.dataset),
        args.audio_dir,
        args.extension,
    )
    tasks = [
        task for task in all_tasks
        if task.task_id in RETRY_TASK_IDS
    ]

    found = {task.task_id for task in tasks}
    missing = RETRY_TASK_IDS - found
    if missing:
        raise ValueError(f"Retry task IDs not found: {sorted(missing)}")

    print("Retrying:", sorted(found), flush=True)

    backend = QwenBackend(args.model_path)

    def generate_with_correction(task, system_prompt, prompt):
        return backend(
            task,
            system_prompt,
            prompt + "\n\n" + FORMAT_CORRECTION,
        )

    run_tasks(
        tasks=tasks,
        model_name="qwen2-audio-7b-instruct",
        output_path=args.output,
        generate=generate_with_correction,
        resume=True,
    )


if __name__ == "__main__":
    main()

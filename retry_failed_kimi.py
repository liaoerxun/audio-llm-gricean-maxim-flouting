#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import build_tasks, load_items, run_tasks
from run_kimi import KimiBackend


FORMAT_CORRECTION = """
IMPORTANT FORMAT CORRECTION:
Your response must contain all four required fields.
Do not answer with only "literal" or "implicature".
Return exactly one complete JSON object in this form:
{"reading":"literal|implicature","maxim":"quality|quantity|manner|none","confidence":0.0,"auditory_cue":"none|pitch|stress|timing|pause|speech_rate|voice_quality|fillers|multiple"}
Output the JSON object only.
"""


def failed_task_ids(path: Path) -> set[str]:
    failed: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)
            if (
                row.get("status") != "ok"
                or row.get("prediction") is None
                or row.get("validation_warnings")
            ):
                failed.add(str(row["task_id"]))

    return failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry only Kimi tasks whose first output could not be parsed."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path("outputs/kimi_results.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kimi_retry_results.jsonl"),
    )
    parser.add_argument(
        "--model-path",
        default="moonshotai/Kimi-Audio-7B-Instruct",
    )
    parser.add_argument("--extension", default="wav")
    args = parser.parse_args()

    failed = failed_task_ids(args.source_results)
    all_tasks = build_tasks(
        load_items(args.dataset),
        args.audio_dir,
        args.extension,
    )
    tasks = [task for task in all_tasks if task.task_id in failed]

    print("Failed task IDs:", sorted(failed), flush=True)
    print("Retry task count:", len(tasks), flush=True)

    if not tasks:
        print("No failed tasks to retry.", flush=True)
        return

    backend = KimiBackend(args.model_path)

    def generate_with_correction(task, system_prompt, prompt):
        return backend(
            task,
            system_prompt,
            prompt + "\n\n" + FORMAT_CORRECTION,
        )

    run_tasks(
        tasks=tasks,
        model_name="kimi-audio-7b-instruct",
        output_path=args.output,
        generate=generate_with_correction,
        resume=True,
    )


if __name__ == "__main__":
    main()

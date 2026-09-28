#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import build_tasks, load_items, user_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pilot tasks without loading a model.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--extension", default="wav")
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot_tasks.jsonl"))
    args = parser.parse_args()
    tasks = build_tasks(load_items(args.dataset), args.audio_dir, args.extension)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps({
                "task_id": task.task_id,
                "item_id": task.item_id,
                "condition": task.condition,
                "gold_maxim": task.maxim,
                "audio_path": str(task.audio_path) if task.audio_path else None,
                "prompt": user_prompt(task),
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()


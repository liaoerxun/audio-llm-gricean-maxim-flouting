#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path

from common import build_tasks, dataset_fingerprint, load_items


def audio_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return {
            "channels": wav.getnchannels(),
            "sample_rate": rate,
            "sample_width_bytes": wav.getsampwidth(),
            "duration_seconds": round(frames / rate, 3) if rate else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pilot YAML and paired WAV files.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--extension", default="wav")
    parser.add_argument("--report", type=Path, default=Path("outputs/pilot_validation.json"))
    args = parser.parse_args()

    items = load_items(args.dataset)
    tasks = build_tasks(items, args.audio_dir, args.extension)
    audio_tasks = [task for task in tasks if task.audio_path is not None]
    errors: list[str] = []
    files: list[dict] = []
    for task in audio_tasks:
        path = task.audio_path
        if not path or not path.is_file():
            errors.append(f"Missing: {path}")
            continue
        try:
            info = audio_info(path)
        except (wave.Error, EOFError) as exc:
            errors.append(f"Unreadable PCM WAV: {path}: {exc}")
            continue
        if info["channels"] != 1:
            errors.append(f"Not mono: {path} ({info['channels']} channels)")
        if info["sample_width_bytes"] != 2:
            errors.append(f"Not 16-bit PCM: {path} ({info['sample_width_bytes'] * 8}-bit)")
        if info["sample_rate"] not in {44100, 48000}:
            errors.append(f"Unexpected sample rate: {path} ({info['sample_rate']} Hz)")
        files.append({"task_id": task.task_id, "path": str(path), **info})

    report = {
        "status": "passed" if not errors else "failed",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_fingerprint(args.dataset),
        "item_count": len(items),
        "task_count": len(tasks),
        "audio_file_count": len(files),
        "maxim_counts": dict(Counter(task.maxim for task in tasks[::3])),
        "errors": errors,
        "files": files,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


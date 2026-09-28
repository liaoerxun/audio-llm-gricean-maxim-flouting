#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stepaudio2 import StepAudio2

from common import Task, build_tasks, load_items, run_tasks


class StepAudioBackend:
    def __init__(self, model_path: str) -> None:
        self.model = StepAudio2(model_path)

    def __call__(self, task: Task, system_prompt: str, prompt: str) -> str:
        content: list[dict] = []
        if task.audio_path is not None:
            content.append({"type": "audio", "audio": str(task.audio_path)})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": content},
            {"role": "assistant", "content": None},
        ]
        _, text, _ = self.model(
            messages,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
        )
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step-Audio 2 mini on pilot conditions.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--model-path", default="Step-Audio-2-mini")
    parser.add_argument("--output", type=Path, default=Path("outputs/stepaudio_results.jsonl"))
    parser.add_argument("--extension", default="wav")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(load_items(args.dataset), args.audio_dir, args.extension)
    backend = StepAudioBackend(args.model_path)
    run_tasks(
        tasks=tasks,
        model_name="step-audio-2-mini",
        output_path=args.output,
        generate=backend,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stepaudio2 import StepAudio2

from common import Task, build_tasks, load_items, run_tasks
from fewshot_common import (
    audio_targets,
    gold_json,
    select_demonstrations,
    trial_text,
)


class StepAudioFewShotBackend:
    def __init__(
        self,
        model_path: str,
        demonstrations: list[Task],
    ) -> None:
        self.model = StepAudio2(model_path)
        self.demonstrations = demonstrations

    def __call__(
        self,
        task: Task,
        system_prompt: str,
        prompt: str,
    ) -> str:
        if task.audio_path is None:
            raise ValueError("Few-shot follow-up is audio-only")

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        for demonstration in self.demonstrations:
            messages.extend([
                {
                    "role": "human",
                    "content": [
                        {
                            "type": "audio",
                            "audio": str(demonstration.audio_path),
                        },
                        {
                            "type": "text",
                            "text": trial_text(demonstration, False),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": gold_json(demonstration),
                },
            ])

        messages.extend([
            {
                "role": "human",
                "content": [
                    {
                        "type": "audio",
                        "audio": str(task.audio_path),
                    },
                    {
                        "type": "text",
                        "text": trial_text(task, True),
                    },
                ],
            },
            {"role": "assistant", "content": None},
        ])

        _, text, _ = self.model(
            messages,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
        )
        return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument(
        "--pilot-dataset",
        type=Path,
        default=Path("pilot_items.yaml"),
    )
    parser.add_argument(
        "--pilot-audio-dir",
        type=Path,
        default=Path("pilot_audio"),
    )
    parser.add_argument(
        "--model-path",
        default="Step-Audio-2-mini",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fewshot/stepaudio_target_results.jsonl"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    targets = audio_targets(build_tasks(
        load_items(args.dataset), args.audio_dir
    ))
    demonstrations = select_demonstrations(build_tasks(
        load_items(args.pilot_dataset),
        args.pilot_audio_dir,
    ))

    backend = StepAudioFewShotBackend(
        args.model_path,
        demonstrations,
    )

    run_tasks(
        tasks=targets,
        model_name="step-audio-2-mini-fewshot-paired-6",
        output_path=args.output,
        generate=backend,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

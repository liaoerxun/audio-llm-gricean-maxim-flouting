#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kimia_infer.api.kimia import KimiAudio

from common import Task, build_tasks, load_items, run_tasks
from fewshot_common import (
    audio_targets,
    gold_json,
    select_demonstrations,
    trial_text,
)


class KimiFewShotBackend:
    def __init__(
        self,
        model_path: str,
        demonstrations: list[Task],
    ) -> None:
        self.model = KimiAudio(
            model_path=model_path,
            load_detokenizer=False,
        )
        self.demonstrations = demonstrations
        self.sampling = {
            "audio_temperature": 0.0,
            "audio_top_k": 1,
            "text_temperature": 0.0,
            "text_top_k": 1,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

    def __call__(
        self,
        task: Task,
        system_prompt: str,
        prompt: str,
    ) -> str:
        if task.audio_path is None:
            raise ValueError("Few-shot follow-up is audio-only")

        messages = []

        for demonstration in self.demonstrations:
            messages.extend([
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": str(demonstration.audio_path),
                },
                {
                    "role": "user",
                    "message_type": "text",
                    "content": trial_text(demonstration, False),
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": gold_json(demonstration),
                },
            ])

        messages.extend([
            {
                "role": "user",
                "message_type": "audio",
                "content": str(task.audio_path),
            },
            {
                "role": "user",
                "message_type": "text",
                "content": (
                    f"{system_prompt}\n\n"
                    f"{trial_text(task, True)}"
                ),
            },
        ])

        _, text = self.model.generate(
            messages,
            **self.sampling,
            output_type="text",
            max_new_tokens=100,
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
        default="moonshotai/Kimi-Audio-7B-Instruct",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fewshot/kimi_target_results.jsonl"),
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

    backend = KimiFewShotBackend(
        args.model_path,
        demonstrations,
    )

    run_tasks(
        tasks=targets,
        model_name="kimi-audio-7b-instruct-fewshot-paired-6",
        output_path=args.output,
        generate=backend,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

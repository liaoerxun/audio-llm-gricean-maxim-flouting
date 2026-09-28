#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kimia_infer.api.kimia import KimiAudio

from common import Task, build_tasks, load_items, run_tasks


class KimiBackend:
    def __init__(self, model_path: str) -> None:
        # Text output is sufficient; omitting the detokenizer reduces memory use.
        self.model = KimiAudio(model_path=model_path, load_detokenizer=False)
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

    def __call__(self, task: Task, system_prompt: str, prompt: str) -> str:
        text_message = {
            "role": "user",
            "message_type": "text",
            "content": f"{system_prompt}\n\n{prompt}",
        }

        if task.audio_path is None:
            messages = [text_message]
        else:
            # Put audio first so the annotation instruction remains the final
            # user message. This reduces Kimi's tendency to answer the audio
            # directly with a single label.
            messages = [
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": str(task.audio_path),
                },
                text_message,
            ]
        _, text = self.model.generate(
            messages,
            **self.sampling,
            output_type="text",
            max_new_tokens=100
        )
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kimi-Audio on pilot conditions.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--model-path", default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/kimi_results.jsonl"))
    parser.add_argument("--extension", default="wav")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(load_items(args.dataset), args.audio_dir, args.extension)
    backend = KimiBackend(args.model_path)
    run_tasks(
        tasks=tasks,
        model_name="kimi-audio-7b-instruct",
        output_path=args.output,
        generate=backend,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()


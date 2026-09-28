#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from common import (
    SYSTEM_PROMPT,
    Task,
    build_tasks,
    load_items,
    run_tasks,
)


DEMONSTRATION_IDS = ("P001", "P006", "P011")

DEMONSTRATION_OUTPUTS = {
    ("P001", "neutral_audio"): {
        "reading": "literal",
        "maxim": "none",
        "confidence": 0.9,
        "auditory_cue": "none",
    },
    ("P001", "marked_audio"): {
        "reading": "implicature",
        "maxim": "quality",
        "confidence": 0.9,
        "auditory_cue": "multiple",
    },
    ("P006", "neutral_audio"): {
        "reading": "literal",
        "maxim": "none",
        "confidence": 0.9,
        "auditory_cue": "none",
    },
    ("P006", "marked_audio"): {
        "reading": "implicature",
        "maxim": "quantity",
        "confidence": 0.9,
        "auditory_cue": "pause",
    },
    ("P011", "neutral_audio"): {
        "reading": "literal",
        "maxim": "none",
        "confidence": 0.9,
        "auditory_cue": "none",
    },
    ("P011", "marked_audio"): {
        "reading": "implicature",
        "maxim": "manner",
        "confidence": 0.9,
        "auditory_cue": "pause",
    },
}


def trial_text(task: Task, *, include_output_instruction: bool) -> str:
    preceding = (
        task.preceding_utterance
        or "[No preceding utterance was provided.]"
    )

    text = (
        f"Context: {task.context}\n"
        f"Preceding utterance: {preceding}\n"
        "The attached audio contains the target utterance.\n"
        "Use the audible delivery as evidence when it is informative."
    )

    if include_output_instruction:
        from common import OUTPUT_INSTRUCTION
        text += f"\n\n{OUTPUT_INSTRUCTION}"

    return text


def audio_user_message(task: Task, *, include_output_instruction: bool) -> dict:
    if task.audio_path is None:
        raise ValueError(f"{task.task_id} has no audio")

    return {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": str(task.audio_path),
            },
            {
                "type": "text",
                "text": trial_text(
                    task,
                    include_output_instruction=include_output_instruction,
                ),
            },
        ],
    }


class QwenFewShotBackend:
    def __init__(
        self,
        model_path: str,
        demonstration_tasks: list[Task],
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
            )
            .eval()
        )

        self.demonstration_tasks = demonstration_tasks
        self.sampling_rate = (
            self.processor.feature_extractor.sampling_rate
        )

        self.demonstration_audios = [
            self.load_audio(task.audio_path)
            for task in demonstration_tasks
        ]

    def load_audio(self, path: Path | None):
        if path is None:
            raise ValueError("Expected an audio path")

        audio, _ = librosa.load(
            path,
            sr=self.sampling_rate,
            mono=True,
        )
        return audio

    def __call__(
        self,
        task: Task,
        system_prompt: str,
        prompt: str,
    ) -> str:
        if task.audio_path is None:
            raise ValueError(
                "The paired few-shot follow-up includes audio trials only"
            )

        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for demonstration in self.demonstration_tasks:
            key = (
                demonstration.item_id,
                demonstration.condition,
            )

            conversation.append(
                audio_user_message(
                    demonstration,
                    include_output_instruction=False,
                )
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        DEMONSTRATION_OUTPUTS[key],
                        separators=(",", ":"),
                    ),
                }
            )

        conversation.append(
            audio_user_message(
                task,
                include_output_instruction=True,
            )
        )

        formatted_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        target_audio = self.load_audio(task.audio_path)
        audios = self.demonstration_audios + [target_audio]

        inputs = self.processor(
            text=formatted_text,
            audios=audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        generated = generated[:, inputs.input_ids.shape[1]:]

        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def select_demonstrations(
    pilot_tasks: list[Task],
) -> list[Task]:
    task_lookup = {
        (task.item_id, task.condition): task
        for task in pilot_tasks
    }

    ordered_keys = [
        ("P001", "neutral_audio"),
        ("P001", "marked_audio"),
        ("P006", "neutral_audio"),
        ("P006", "marked_audio"),
        ("P011", "neutral_audio"),
        ("P011", "marked_audio"),
    ]

    demonstrations = []

    for key in ordered_keys:
        if key not in task_lookup:
            raise ValueError(
                f"Missing demonstration task: {key}"
            )
        demonstrations.append(task_lookup[key])

    return demonstrations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired multimodal 6-shot Qwen evaluation."
    )
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
        default="Qwen/Qwen2-Audio-7B-Instruct",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/fewshot/qwen_target_results.jsonl"
        ),
    )
    parser.add_argument("--extension", default="wav")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    target_tasks = build_tasks(
        load_items(args.dataset),
        args.audio_dir,
        args.extension,
    )
    target_tasks = [
        task
        for task in target_tasks
        if task.condition in {
            "neutral_audio",
            "marked_audio",
        }
    ]

    pilot_tasks = build_tasks(
        load_items(args.pilot_dataset),
        args.pilot_audio_dir,
        args.extension,
    )
    demonstrations = select_demonstrations(pilot_tasks)

    backend = QwenFewShotBackend(
        args.model_path,
        demonstrations,
    )

    run_tasks(
        tasks=target_tasks,
        model_name=(
            "qwen2-audio-7b-instruct-fewshot-paired-6"
        ),
        output_path=args.output,
        generate=backend,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

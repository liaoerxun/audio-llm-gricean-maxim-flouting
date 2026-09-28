#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from common import Task, build_tasks, load_items, run_tasks


class QwenBackend:
    def __init__(self, model_path: str) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    def __call__(self, task: Task, system_prompt: str, prompt: str) -> str:
        content: list[dict] = []
        audios = None
        if task.audio_path is not None:
            content.append({"type": "audio", "audio_url": str(task.audio_path)})
            audio, _ = librosa.load(
                task.audio_path,
                sr=self.processor.feature_extractor.sampling_rate,
                mono=True,
            )
            audios = [audio]
        content.append({"type": "text", "text": prompt})
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )


        # Qwen2-Audio's multimodal wrapper errors when no audio features exist.
        # For transcript-only trials, use the language-model component from the
        # same checkpoint.
        if audios is None:
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.language_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
        else:
            processor_kwargs ={
                "text": text,
                "audios": audios,
                "return_tensors": "pt",
                "padding": True,
             }

            inputs = self.processor(
                **processor_kwargs
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2-Audio on pilot conditions.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--model-path", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen_results.jsonl"))
    parser.add_argument("--extension", default="wav")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    tasks = build_tasks(load_items(args.dataset), args.audio_dir, args.extension)
    backend = QwenBackend(args.model_path)
    run_tasks(
        tasks=tasks,
        model_name="qwen2-audio-7b-instruct",
        output_path=args.output,
        generate=backend,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

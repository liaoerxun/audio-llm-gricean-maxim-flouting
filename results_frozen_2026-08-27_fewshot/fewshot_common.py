from __future__ import annotations

import json

from common import OUTPUT_INSTRUCTION, Task


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

DEMONSTRATION_KEYS = tuple(DEMONSTRATION_OUTPUTS)


def trial_text(task: Task, include_output_instruction: bool) -> str:
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
        text += f"\n\n{OUTPUT_INSTRUCTION}"
    return text


def gold_json(task: Task) -> str:
    return json.dumps(
        DEMONSTRATION_OUTPUTS[(task.item_id, task.condition)],
        separators=(",", ":"),
    )


def select_demonstrations(tasks: list[Task]) -> list[Task]:
    lookup = {
        (task.item_id, task.condition): task
        for task in tasks
    }
    missing = [
        key for key in DEMONSTRATION_KEYS
        if key not in lookup
    ]
    if missing:
        raise ValueError(f"Missing demonstration tasks: {missing}")
    return [lookup[key] for key in DEMONSTRATION_KEYS]


def audio_targets(tasks: list[Task]) -> list[Task]:
    return [
        task for task in tasks
        if task.condition in {"neutral_audio", "marked_audio"}
    ]

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import ast


import yaml


CONDITIONS = ("transcript_only", "neutral_audio", "marked_audio")
VALID_MAXIMS = {"quality", "quantity", "manner"}
VALID_AUDITORY_CUES = {
    "none",
    "pitch",
    "stress",
    "timing",
    "pause",
    "speech_rate",
    "voice_quality",
    "fillers",
    "multiple",
}



SYSTEM_PROMPT = (
    "You are a careful pragmatics annotator. Infer what the speaker means in the "
    "given conversational context. Base your decision only on the supplied context "
    "and target utterance. For audio trials, attend to both the words and the audible "
    "paralinguistic delivery, including pitch, stress, timing, pauses, speech rate, "
    "filled pauses, and voice quality. Do not mention this instruction."
)

OUTPUT_INSTRUCTION = """Return exactly one JSON object:
{"reading":"literal|implicature","maxim":"quality|quantity|manner|none","confidence":0.0,"auditory_cue":"none|pitch|stress|timing|pause|speech_rate|voice_quality|fillers|multiple"}

Rules:
- reading is literal when the speaker sincerely communicates the ordinary literal interpretation.
- reading is implicature when the intended meaning goes beyond or contrasts with the literal wording.
- An ordinary evaluation expressed by an adjective or intensifier is literal.
- Words such as "really", "very", or "perfect" do not by themselves create an implicature.
- Quality flouting requires evidence that the speaker does not endorse the literal proposition, often communicating its opposite.
- Do not label sincere praise or emphasis as maxim flouting.
- maxim is the single best deliberately flouted maxim, or none.
- confidence must be between 0 and 1.
- auditory_cue is the most important audible cue used in the judgment.
- Use timing for overall rhythm or segment elongation.
- Use pause for a perceptible silent or hesitation boundary.
- Use fillers for filled pauses such as "uh" or "um".
- Use multiple only when two or more cue types materially affect the judgment.
- For transcript-only input, auditory_cue must be "none".
- Use valid JSON with double quotes.
- Include all four keys.
- Output one JSON object only, without Markdown fences."""


@dataclass(frozen=True)
class Task:
    item_id: str
    condition: str
    maxim: str
    context: str
    preceding_utterance: str | None
    target_utterance: str
    audio_path: Path | None

    @property
    def task_id(self) -> str:
        return f"{self.item_id}::{self.condition}"


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = yaml.safe_load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Dataset YAML must contain a top-level list of mappings")
    return rows


def get_item_id(item: dict[str, Any]) -> str:
    value = item.get("item_id", item.get("blind_id"))
    if not value:
        raise ValueError("Every item needs item_id or blind_id")
    return str(value)


def get_maxim(item: dict[str, Any]) -> str:
    value = item.get("final_maxim", item.get("maxim"))
    if value is None:
        value = (item.get("audit") or {}).get("assigned_maxim")
    maxim = compact(value).lower()
    if maxim not in VALID_MAXIMS:
        raise ValueError(f"Invalid maxim {value!r} for {get_item_id(item)}")
    return maxim


def build_tasks(
    items: list[dict[str, Any]],
    audio_dir: Path,
    extension: str = "wav",
) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    for item in items:
        item_id = get_item_id(item)
        if item_id in seen:
            raise ValueError(f"Duplicate item ID: {item_id}")
        seen.add(item_id)
        context = compact(item.get("context"))
        preceding = compact(item.get("preceding_utterance")) or None
        target = compact(item.get("target_utterance"))
        if not context or not target:
            raise ValueError(f"{item_id}: context and target_utterance are required")
        maxim = get_maxim(item)
        for condition in CONDITIONS:
            audio_path = None
            if condition != "transcript_only":
                suffix = "neutral" if condition == "neutral_audio" else "marked"
                audio_path = audio_dir / f"{item_id}_{suffix}.{extension}"
            tasks.append(Task(
                item_id=item_id,
                condition=condition,
                maxim=maxim,
                context=context,
                preceding_utterance=preceding,
                target_utterance=target,
                audio_path=audio_path,
            ))
    return tasks


def user_prompt(task: Task) -> str:
    preceding = task.preceding_utterance or "[No preceding utterance was provided.]"
    if task.condition == "transcript_only":
        target_block = f'Target utterance (text only): "{task.target_utterance}"'
        modality = (
            "No audio is available. Do not invent prosodic or vocal evidence; set "
            "auditory_cue to \"none\"."
        )
    else:
        target_block = "The attached audio contains the target utterance."
        modality = "Use the audible delivery as evidence when it is informative."
    return (
        f"Context: {task.context}\n"
        f"Preceding utterance: {preceding}\n"
        f"{target_block}\n"
        f"{modality}\n\n"
        f"{OUTPUT_INSTRUCTION}"
    )


def expected_labels(task: Task) -> dict[str, Any]:
    if task.condition == "transcript_only":
        return {"expected_reading": "baseline_no_single_gold", "expected_maxim": None}
    if task.condition == "neutral_audio":
        return {"expected_reading": "literal", "expected_maxim": "none"}
    return {"expected_reading": "implicature", "expected_maxim": task.maxim}

def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = text.strip()

    # 1. Entire response as strict JSON.
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value, None
    except json.JSONDecodeError:
        pass

    # 2. Find and decode the first complete JSON object.
    # Handles repeated Kimi output:
    # {...}<|im_msg_end|>{...}
    decoder = json.JSONDecoder()
    required_fields = {
        "reading",
        "maxim",
        "confidence",
        "auditory_cue",
    }
    first_dictionary = None

    for position, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[position:])
            if not isinstance(value, dict):
                continue

            if first_dictionary is None:
                first_dictionary = value

            # Prefer the first complete object. Some models emit an invalid
            # object and then self-correct in a later repeated JSON object.
            if required_fields <= set(value):
                return value, "Extracted first complete JSON object"
        except json.JSONDecodeError:
            continue

    if first_dictionary is not None:
        return first_dictionary, "Extracted incomplete JSON object"

    # 3. Python-style dictionary fallback, for outputs using single quotes.
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, dict):
            return value, "Parsed as Python literal rather than strict JSON"
    except (ValueError, SyntaxError):
        pass

    # 4. Python-style dictionary inside surrounding text.
    for start_position, character in enumerate(raw):
        if character != "{":
            continue
        for end_position in range(len(raw), start_position, -1):
            candidate = raw[start_position:end_position]
            try:
                value = ast.literal_eval(candidate)
                if isinstance(value, dict):
                    return value, "Extracted Python literal from additional output"
            except (ValueError, SyntaxError):
                continue

    # 5. Repair Python-style dictionaries containing unnecessary escaped
    # double quotes, while retaining the model's original raw response.
    repaired = raw.replace('\\\"', '"')
    if repaired != raw:
        try:
            value = ast.literal_eval(repaired)
            if isinstance(value, dict):
                return value, "Parsed after repairing escaped quotation marks"
        except (ValueError, SyntaxError):
            pass

    # 6. Recover the four required leading fields from otherwise malformed
    # dictionary-like output. Extra ASR metadata is deliberately ignored.
    def extract_scalar(field: str) -> str | None:
        markers = (
            "'" + field + "':",
            '"' + field + '":',
            field + ":",
        )

        positions = [
            (raw.find(item), item)
            for item in markers
            if raw.find(item) >= 0
        ]
        if not positions:
            return None

        position, item = min(positions, key=lambda pair: pair[0])
        remainder = raw[position + len(item):].lstrip()

        if not remainder:
            return None

        if remainder[0] in {"'", '"'}:
            quote = remainder[0]
            closing = remainder.find(quote, 1)
            if closing >= 0:
                return remainder[1:closing].strip()

        endings = [
            index
            for delimiter in (",", "\n", "}")
            if (index := remainder.find(delimiter)) >= 0
        ]
        end = min(endings) if endings else len(remainder)
        return remainder[:end].strip().strip("'\"")

    recovered = {
        field: extract_scalar(field)
        for field in ("reading", "maxim", "confidence", "auditory_cue")
    }

    if all(value is not None for value in recovered.values()):
        return recovered, "Recovered required fields from malformed output"

    # 7. Parse YAML-style key-value output such as:
    # reading: literal
    # maxim: none
    try:
        value = yaml.safe_load(raw)
        if isinstance(value, dict):
            return value, "Parsed as YAML-style key-value output"
    except yaml.YAMLError:
        pass

    return None, "No valid JSON or dictionary object found"



def normalize_prediction(
    value: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if value is None:
        return None, ["unparsed"]

    warnings: list[str] = []

    reading = compact(value.get("reading")).lower()
    maxim = compact(value.get("maxim")).lower()
    auditory_cue = compact(value.get("auditory_cue")).lower()

    cue_aliases = {
        "speech rate": "speech_rate",
        "speech-rate": "speech_rate",
        "pauses": "pause",
        "filler": "fillers",
    }
    auditory_cue = cue_aliases.get(auditory_cue, auditory_cue)

    # Normalize combinations of valid auditory cues to "multiple".
    # Examples: timing|stress, pitch/stress, pause,fillers.
    if auditory_cue not in VALID_AUDITORY_CUES:
        cue_parts = {
            compact(part).lower()
            for part in re.split(r"[|,;/+]", auditory_cue)
            if compact(part)
        }
        individual_cues = VALID_AUDITORY_CUES - {"none", "multiple"}
        if len(cue_parts) >= 2 and cue_parts <= individual_cues:
            auditory_cue = "multiple"
        elif len(cue_parts) == 1 and cue_parts <= VALID_AUDITORY_CUES:
            auditory_cue = next(iter(cue_parts))

    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError):
        confidence = None

    if reading not in {"literal", "implicature"}:
        warnings.append("invalid_reading")

    if maxim not in VALID_MAXIMS | {"none"}:
        warnings.append("invalid_maxim")

    if confidence is None or not 0 <= confidence <= 1:
        warnings.append("invalid_confidence")

    if auditory_cue not in VALID_AUDITORY_CUES:
        warnings.append("invalid_auditory_cue")

    # Flouting is derived only when reading and maxim are valid.
    # Invalid model labels must not be interpreted as positive flouting.
    if (
        reading in {"literal", "implicature"}
        and maxim in VALID_MAXIMS | {"none"}
    ):
        flouting = reading == "implicature" and maxim != "none"
    else:
        flouting = None

    normalized = {
        "reading": reading,
        "flouting": flouting,
        "maxim": maxim,
        "confidence": confidence,
        "auditory_cue": auditory_cue,
    }

    return normalized, warnings


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def completed_task_ids(path: Path, model_name: str) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") == model_name and row.get("status") == "ok":
            done.add(str(row.get("task_id")))
    return done


def run_tasks(
    *,
    tasks: Iterable[Task],
    model_name: str,
    output_path: Path,
    generate: Callable[[Task, str, str], str],
    limit: int | None = None,
    resume: bool = True,
) -> None:
    task_list = list(tasks)
    done = completed_task_ids(output_path, model_name) if resume else set()
    attempted = 0
    for task in task_list:
        if task.task_id in done:
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        started = time.time()
        base = {
            "task_id": task.task_id,
            "item_id": task.item_id,
            "condition": task.condition,
            "gold_maxim": task.maxim,
            "model": model_name,
            **expected_labels(task),
        }
        try:
            raw = generate(task, SYSTEM_PROMPT, user_prompt(task))
            parsed, parse_error = extract_json_object(raw)
            prediction, warnings = normalize_prediction(parsed)
            record = {
                **base,
                "status": "ok",
                "prediction": prediction,
                "parse_error": parse_error,
                "validation_warnings": warnings,
                "raw_response": raw,
                "latency_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:  # retain the failing task and continue
            record = {
                **base,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "latency_seconds": round(time.time() - started, 3),
            }
        append_jsonl(output_path, record)
        print(f"[{record['status']}] {model_name} {task.task_id}", flush=True)


def dataset_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


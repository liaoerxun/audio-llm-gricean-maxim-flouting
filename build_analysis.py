#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


FROZEN = Path("results_frozen_2026-08-20")
OUTPUT = Path("analysis")

FILES = {
    "qwen2-audio-7b-instruct": FROZEN / "qwen_target_results.jsonl",
    "kimi-audio-7b-instruct": FROZEN / "kimi_target_results.jsonl",
    "step-audio-2-mini": FROZEN / "stepaudio_target_results.jsonl",
}

VALID_READINGS = {"literal", "implicature"}
VALID_MAXIMS = {"quality", "quantity", "manner", "none"}
VALID_CUES = {
    "none", "pitch", "stress", "timing", "pause",
    "speech_rate", "voice_quality", "fillers", "multiple",
}


def optional_bool(value):
    if value is None:
        return ""
    return int(bool(value))


rows = []

for model, path in FILES.items():
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        source = json.loads(line)
        prediction = source.get("prediction") or {}
        condition = source["condition"]

        reading = prediction.get("reading")
        maxim = prediction.get("maxim")
        cue = prediction.get("auditory_cue")
        confidence = prediction.get("confidence")

        reading_valid = reading in VALID_READINGS
        maxim_valid = maxim in VALID_MAXIMS
        cue_valid = cue in VALID_CUES
        warnings = source.get("validation_warnings", [])

        format_valid = (
            source.get("status") == "ok"
            and reading_valid
            and maxim_valid
            and isinstance(confidence, (int, float))
            and 0 <= confidence <= 1
            and cue_valid
            and not warnings
        )

        if condition == "transcript_only":
            expected_reading = None
            expected_maxim = None
        elif condition == "neutral_audio":
            expected_reading = "literal"
            expected_maxim = "none"
        else:
            expected_reading = "implicature"
            expected_maxim = source["gold_maxim"]

        reading_correct = (
            reading == expected_reading
            if expected_reading is not None and reading_valid
            else None
        )
        maxim_correct = (
            maxim == expected_maxim
            if expected_maxim is not None and maxim_valid
            else None
        )
        joint_correct = (
            reading_correct and maxim_correct
            if reading_correct is not None and maxim_correct is not None
            else None
        )

        rows.append({
            "model": model,
            "task_id": source["task_id"],
            "item_id": source["item_id"],
            "condition": condition,
            "gold_maxim": source["gold_maxim"],
            "expected_reading": expected_reading or "",
            "expected_maxim": expected_maxim or "",
            "reading": reading or "",
            "flouting": (
                "" if prediction.get("flouting") is None
                else int(prediction["flouting"])
            ),
            "maxim": maxim or "",
            "confidence": confidence if confidence is not None else "",
            "auditory_cue": cue or "",
            "reading_valid": int(reading_valid),
            "maxim_valid": int(maxim_valid),
            "format_valid": int(format_valid),
            "reading_correct": optional_bool(reading_correct),
            "maxim_correct": optional_bool(maxim_correct),
            "joint_correct": optional_bool(joint_correct),
            "retry_used": int(bool(source.get("retry_used", False))),
            "retry_outcome": source.get("retry_outcome", ""),
            "validation_warnings": "|".join(warnings),
        })

rows.sort(key=lambda row: (
    row["model"],
    row["item_id"],
    row["condition"],
))

OUTPUT.mkdir(exist_ok=True)

fieldnames = list(rows[0])
with (OUTPUT / "analysis_long.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# Condition-level summary
groups = defaultdict(list)
for row in rows:
    groups[(row["model"], row["condition"])].append(row)

summary_rows = []

for (model, condition), group in sorted(groups.items()):
    reading_valid_rows = [r for r in group if r["reading_valid"]]
    scored_reading = [
        r for r in group if r["reading_correct"] != ""
    ]
    scored_maxim = [
        r for r in group if r["maxim_correct"] != ""
    ]
    scored_joint = [
        r for r in group if r["joint_correct"] != ""
    ]
    confidences = [
        float(r["confidence"])
        for r in group
        if r["confidence"] != ""
    ]

    summary_rows.append({
        "model": model,
        "condition": condition,
        "n": len(group),
        "format_valid_n": sum(r["format_valid"] for r in group),
        "format_valid_rate": sum(r["format_valid"] for r in group) / len(group),
        "implicature_rate": (
            sum(r["reading"] == "implicature" for r in reading_valid_rows)
            / len(reading_valid_rows)
            if reading_valid_rows else ""
        ),
        "reading_accuracy": (
            sum(r["reading_correct"] for r in scored_reading)
            / len(scored_reading)
            if scored_reading else ""
        ),
        "maxim_accuracy": (
            sum(r["maxim_correct"] for r in scored_maxim)
            / len(scored_maxim)
            if scored_maxim else ""
        ),
        "joint_accuracy": (
            sum(r["joint_correct"] for r in scored_joint)
            / len(scored_joint)
            if scored_joint else ""
        ),
        "mean_confidence": mean(confidences) if confidences else "",
    })

with (OUTPUT / "condition_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(summary_rows[0]),
    )
    writer.writeheader()
    writer.writerows(summary_rows)


# Paired neutral/marked contrast
by_item = defaultdict(dict)
for row in rows:
    by_item[(row["model"], row["item_id"])][row["condition"]] = row

paired_by_model = defaultdict(list)

for (model, item_id), conditions in by_item.items():
    neutral = conditions.get("neutral_audio")
    marked = conditions.get("marked_audio")
    if not neutral or not marked:
        continue
    if not neutral["reading_valid"] or not marked["reading_valid"]:
        continue

    paired_by_model[model].append(
        neutral["reading"] == "literal"
        and marked["reading"] == "implicature"
    )

paired_rows = []

for model, values in sorted(paired_by_model.items()):
    paired_rows.append({
        "model": model,
        "paired_items": len(values),
        "correct_neutral_marked_contrast_n": sum(values),
        "correct_neutral_marked_contrast_rate": (
            sum(values) / len(values) if values else ""
        ),
    })

with (OUTPUT / "paired_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(paired_rows[0]),
    )
    writer.writeheader()
    writer.writerows(paired_rows)

print("analysis_long.csv rows:", len(rows))
print("condition_summary.csv rows:", len(summary_rows))
print("paired_summary.csv rows:", len(paired_rows))

#!/usr/bin/env python3
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


INPUT = Path("analysis/analysis_long.csv")
OUTPUT = Path("analysis")

with INPUT.open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 1. Results stratified by gold maxim and condition
groups = defaultdict(list)

for row in rows:
    groups[
        (row["model"], row["condition"], row["gold_maxim"])
    ].append(row)

stratified = []

for (model, condition, gold_maxim), group in sorted(groups.items()):
    valid_readings = [r for r in group if r["reading_valid"] == "1"]
    scored_readings = [r for r in group if r["reading_correct"] != ""]
    scored_maxims = [r for r in group if r["maxim_correct"] != ""]

    stratified.append({
        "model": model,
        "condition": condition,
        "gold_maxim": gold_maxim,
        "n": len(group),
        "implicature_rate": (
            sum(r["reading"] == "implicature" for r in valid_readings)
            / len(valid_readings)
            if valid_readings else ""
        ),
        "reading_accuracy": (
            sum(int(r["reading_correct"]) for r in scored_readings)
            / len(scored_readings)
            if scored_readings else ""
        ),
        "maxim_accuracy": (
            sum(int(r["maxim_correct"]) for r in scored_maxims)
            / len(scored_maxims)
            if scored_maxims else ""
        ),
    })

write_csv(OUTPUT / "stratified_summary.csv", stratified)


# 2. Predicted-maxim distributions
maxim_rows = []

for (model, condition), group in sorted(
    defaultdict(list, {
        key: [
            r for r in rows
            if (r["model"], r["condition"]) == key
        ]
        for key in {
            (r["model"], r["condition"]) for r in rows
        }
    }).items()
):
    counts = Counter(r["maxim"] or "INVALID" for r in group)

    for predicted_maxim in sorted(counts):
        maxim_rows.append({
            "model": model,
            "condition": condition,
            "predicted_maxim": predicted_maxim,
            "count": counts[predicted_maxim],
            "rate": counts[predicted_maxim] / len(group),
        })

write_csv(OUTPUT / "predicted_maxim_distribution.csv", maxim_rows)


# 3. Auditory-cue distribution for audio conditions
cue_groups = defaultdict(list)

for row in rows:
    if row["condition"] != "transcript_only":
        cue_groups[(row["model"], row["condition"])].append(row)

cue_rows = []

for (model, condition), group in sorted(cue_groups.items()):
    counts = Counter(r["auditory_cue"] or "INVALID" for r in group)

    for cue in sorted(counts):
        cue_rows.append({
            "model": model,
            "condition": condition,
            "auditory_cue": cue,
            "count": counts[cue],
            "rate": counts[cue] / len(group),
        })

write_csv(OUTPUT / "auditory_cue_distribution.csv", cue_rows)


# 4. Exact paired neutral-to-marked shift test
by_item = defaultdict(dict)

for row in rows:
    by_item[(row["model"], row["item_id"])][row["condition"]] = row


def exact_two_sided_binomial(successes, trials):
    if trials == 0:
        return 1.0

    probability = 0.0
    observed_probability = math.comb(trials, successes) / (2 ** trials)

    for value in range(trials + 1):
        candidate = math.comb(trials, value) / (2 ** trials)
        if candidate <= observed_probability + 1e-15:
            probability += candidate

    return min(1.0, probability)


models = sorted({row["model"] for row in rows})
paired_tests = []

for model in models:
    literal_to_implicature = 0
    implicature_to_literal = 0
    unchanged = 0
    usable = 0

    for (item_model, _), conditions in by_item.items():
        if item_model != model:
            continue

        neutral = conditions.get("neutral_audio")
        marked = conditions.get("marked_audio")

        if not neutral or not marked:
            continue
        if neutral["reading_valid"] != "1" or marked["reading_valid"] != "1":
            continue

        usable += 1
        pair = (neutral["reading"], marked["reading"])

        if pair == ("literal", "implicature"):
            literal_to_implicature += 1
        elif pair == ("implicature", "literal"):
            implicature_to_literal += 1
        else:
            unchanged += 1

    discordant = literal_to_implicature + implicature_to_literal
    p_value = exact_two_sided_binomial(
        literal_to_implicature,
        discordant,
    )

    paired_tests.append({
        "model": model,
        "usable_pairs": usable,
        "literal_to_implicature": literal_to_implicature,
        "implicature_to_literal": implicature_to_literal,
        "unchanged": unchanged,
        "discordant_pairs": discordant,
        "exact_mcnemar_p": p_value,
    })

write_csv(OUTPUT / "paired_shift_tests.csv", paired_tests)

print("Created:")
print(" analysis/stratified_summary.csv")
print(" analysis/predicted_maxim_distribution.csv")
print(" analysis/auditory_cue_distribution.csv")
print(" analysis/paired_shift_tests.csv")

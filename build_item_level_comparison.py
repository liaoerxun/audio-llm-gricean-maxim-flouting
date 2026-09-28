#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import yaml


DATASET = Path("target_items.yaml")
PREDICTIONS = Path("analysis/analysis_long.csv")
OUTPUT = Path("analysis/item_level_comparison.csv")


items = yaml.safe_load(
    DATASET.read_text(encoding="utf-8")
)

item_rows = []

for item in items:
    item_id = str(
        item.get("item_id", item.get("blind_id"))
    )

    gold_maxim = item.get(
        "final_maxim",
        item.get("maxim"),
    )

    if gold_maxim is None:
        gold_maxim = (
            item.get("audit") or {}
        ).get("assigned_maxim")

    item_rows.append({
        "item_id": item_id,
        "gold_maxim_from_dataset": gold_maxim,
        "context": item.get("context", ""),
        "preceding_utterance": item.get(
            "preceding_utterance", ""
        ),
        "target_utterance": item.get(
            "target_utterance", ""
        ),
    })

item_df = pd.DataFrame(item_rows)

predictions = pd.read_csv(PREDICTIONS)

fields = [
    "reading",
    "maxim",
    "auditory_cue",
    "confidence",
    "format_valid",
]

wide = predictions.pivot(
    index=["model", "item_id", "gold_maxim"],
    columns="condition",
    values=fields,
)

wide.columns = [
    f"{field}_{condition}"
    for field, condition in wide.columns
]

wide = wide.reset_index()

result = wide.merge(
    item_df,
    on="item_id",
    how="left",
    validate="many_to_one",
)

result["neutral_marked_transition"] = (
    result["reading_neutral_audio"].fillna("missing")
    + "_to_"
    + result["reading_marked_audio"].fillna("missing")
)

result["correct_contrast"] = (
    (result["reading_neutral_audio"] == "literal")
    & (result["reading_marked_audio"] == "implicature")
)

result["reading_invariant"] = (
    result["reading_neutral_audio"]
    == result["reading_marked_audio"]
)

result["marked_reading_maxim_consistent"] = (
    (
        (result["reading_marked_audio"] == "literal")
        & (result["maxim_marked_audio"] == "none")
    )
    |
    (
        (result["reading_marked_audio"] == "implicature")
        & result["maxim_marked_audio"].isin(
            ["quality", "quantity", "manner"]
        )
    )
)

result = result.sort_values(
    ["model", "gold_maxim", "item_id"]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("Rows:", len(result))
print("Models:", result["model"].nunique())
print("Items per model:")
print(result.groupby("model")["item_id"].nunique())
print("\nCorrect contrasts:")
print(result.groupby("model")["correct_contrast"].sum())
print("\nCreated:", OUTPUT)

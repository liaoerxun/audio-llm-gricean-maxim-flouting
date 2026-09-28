#!/usr/bin/env python3
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("analysis")
OUTPUT = INPUT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "kimi-audio-7b-instruct",
    "qwen2-audio-7b-instruct",
    "step-audio-2-mini",
]

MODEL_LABELS = {
    "kimi-audio-7b-instruct": "Kimi-Audio",
    "qwen2-audio-7b-instruct": "Qwen2-Audio",
    "step-audio-2-mini": "Step-Audio 2",
}

CONDITION_ORDER = [
    "transcript_only",
    "neutral_audio",
    "marked_audio",
]

CONDITION_LABELS = {
    "transcript_only": "Transcript",
    "neutral_audio": "Neutral",
    "marked_audio": "Marked",
}

COLORS = {
    "transcript_only": "#9CA3AF",
    "neutral_audio": "#4C78A8",
    "marked_audio": "#E45756",
}


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return (np.nan, np.nan)

    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return centre - margin, centre + margin


def save(fig, name):
    fig.savefig(
        OUTPUT / f"{name}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Figure 1: implicature rate by condition
data = pd.read_csv(INPUT / "analysis_long.csv")
data = data[data["reading_valid"] == 1]

fig, ax = plt.subplots(figsize=(8.5, 4.8))
width = 0.23
x = np.arange(len(MODEL_ORDER))

for index, condition in enumerate(CONDITION_ORDER):
    values = []
    lower_errors = []
    upper_errors = []

    for model in MODEL_ORDER:
        subset = data[
            (data["model"] == model)
            & (data["condition"] == condition)
        ]
        successes = (subset["reading"] == "implicature").sum()
        total = len(subset)
        rate = successes / total
        lower, upper = wilson_interval(successes, total)

        values.append(rate)
        lower_errors.append(rate - lower)
        upper_errors.append(upper - rate)

    positions = x + (index - 1) * width

    ax.bar(
        positions,
        values,
        width,
        label=CONDITION_LABELS[condition],
        color=COLORS[condition],
        yerr=[lower_errors, upper_errors],
        capsize=3,
        edgecolor="white",
    )

ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Implicature prediction rate")
ax.set_title("Predicted implicature rates across input conditions")
ax.legend(frameon=False, ncol=3, loc="upper center")
ax.axhline(0.5, color="#BBBBBB", linestyle="--", linewidth=0.8)

fig.tight_layout()
save(fig, "implicature_rate_by_condition")


# Figure 2: accuracy and reading–maxim consistency
condition = pd.read_csv(INPUT / "condition_summary.csv")
consistency = pd.read_csv(
    INPUT / "reading_maxim_consistency.csv"
)

merged = condition.merge(
    consistency[
        ["model", "condition", "consistency_rate"]
    ],
    on=["model", "condition"],
)

merged = merged[
    merged["condition"].isin(
        ["neutral_audio", "marked_audio"]
    )
]

metrics = [
    ("reading_accuracy", "Reading"),
    ("maxim_accuracy", "Maxim"),
    ("joint_accuracy", "Joint"),
    ("consistency_rate", "Consistency"),
]

fig, axes = plt.subplots(
    1, 3,
    figsize=(12, 4.2),
    sharey=True,
)

for ax, model in zip(axes, MODEL_ORDER):
    subset = merged[merged["model"] == model]
    metric_x = np.arange(len(metrics))
    width = 0.34

    for offset, condition_name in [
        (-width / 2, "neutral_audio"),
        (width / 2, "marked_audio"),
    ]:
        row = subset[
            subset["condition"] == condition_name
        ].iloc[0]

        values = [row[column] for column, _ in metrics]

        ax.bar(
            metric_x + offset,
            values,
            width,
            color=COLORS[condition_name],
            label=CONDITION_LABELS[condition_name],
        )

    ax.set_title(MODEL_LABELS[model])
    ax.set_xticks(metric_x)
    ax.set_xticklabels(
        [label for _, label in metrics],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Proportion")
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle(
    "Accuracy and reading–maxim consistency",
    y=1.02,
)
fig.tight_layout()
save(fig, "accuracy_and_consistency")


# Figure 3: paired neutral-to-marked transitions
paired = pd.read_csv(INPUT / "paired_shift_tests.csv")
paired = paired.set_index("model").loc[MODEL_ORDER]

transition_columns = [
    ("literal_to_literal", "Literal → Literal", "#9CA3AF"),
    (
        "literal_to_implicature",
        "Literal → Implicature",
        "#59A14F",
    ),
    (
        "implicature_to_implicature",
        "Implicature → Implicature",
        "#E45756",
    ),
    (
        "implicature_to_literal",
        "Implicature → Literal",
        "#F2CF5B",
    ),
]

fig, ax = plt.subplots(figsize=(8.5, 4.8))
left = np.zeros(len(MODEL_ORDER))

for column, label, color in transition_columns:
    values = paired[column].to_numpy()
    ax.barh(
        [MODEL_LABELS[m] for m in MODEL_ORDER],
        values,
        left=left,
        label=label,
        color=color,
    )
    left += values

ax.set_xlim(0, 45)
ax.set_xlabel("Number of paired items")
ax.set_title("Neutral-to-marked reading transitions")
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
)

fig.tight_layout()
save(fig, "paired_transitions")

print("Figures written to:", OUTPUT)
for path in sorted(OUTPUT.iterdir()):
    print(path)

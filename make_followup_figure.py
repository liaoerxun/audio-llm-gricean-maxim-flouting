#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT = Path("analysis/followup/prompting_summary.csv")
OUTPUT_DIR = Path("analysis/followup/figures")

REGIMES = [
    "zero_shot",
    "paired_6shot",
    "prosody_informed",
]

REGIME_LABELS = [
    "Zero-shot",
    "Paired\n6-shot",
    "Prosody-\ninformed",
]

MODELS = ["Qwen", "Kimi", "Step-Audio"]

COLORS = {
    "Qwen": "#6B7280",
    "Kimi": "#C44E52",
    "Step-Audio": "#0C2577",
}

MARKERS = {
    "Qwen": "o",
    "Kimi": "s",
    "Step-Audio": "^",
}


def load_rows():
    with INPUT.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows, model):
    lookup = {
        (row["model"], row["regime"]): float(row["correct_reading_pair_rate"])
        for row in rows
    }

    return [100 * lookup[(model, regime)] for regime in REGIMES]


def main():
    rows = load_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
        }
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.0),
        constrained_layout=True,
    )

    x = range(len(REGIMES))

    for model in MODELS:
        y = values(rows, model)

        ax.plot(
            x,
            y,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=10,
            linewidth=3,
            label=model,
        )

        for x_value, y_value in zip(x, y):
            ax.annotate(
                f"{y_value:.1f}",
                (x_value, y_value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                fontweight=("bold" if model == "Kimi" else "normal"),
                color=COLORS[model],
            )

    ax.set_xticks(list(x), REGIME_LABELS)
    ax.set_ylim(-4, 74)
    ax.set_ylabel("Correct neutral→marked pairs (%)")
    ax.set_title(
        "Condition-sensitive reading across prompting regimes",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right",
        frameon=False,
    )

    fig.savefig(
        OUTPUT_DIR / "prompting_regime_comparison.svg",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "prompting_regime_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()

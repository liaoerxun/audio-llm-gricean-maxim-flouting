#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine JSONL model outputs into CSV and summary JSON.")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("outputs/pilot_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/pilot_summary.json"))
    args = parser.parse_args()

    rows = []
    for path in args.results:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pred = row.get("prediction") or {}
            rows.append({
                "task_id": row.get("task_id"),
                "item_id": row.get("item_id"),
                "model": row.get("model"),
                "condition": row.get("condition"),
                "gold_maxim": row.get("gold_maxim"),
                "expected_reading": row.get("expected_reading"),
                "expected_maxim": row.get("expected_maxim"),
                "status": row.get("status"),
                "predicted_reading": pred.get("reading"),
                "predicted_flouting": pred.get("flouting"),
                "predicted_maxim": pred.get("maxim"),
                "confidence": pred.get("confidence"),
                "auditory_cue": pred.get("auditory_cue"),
                "parse_warnings": ";".join(row.get("validation_warnings") or []),
                "latency_seconds": row.get("latency_seconds"),
                "raw_response": row.get("raw_response"),
                "error": row.get("error"),
            })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["task_id"]
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["status"] == "ok"]
    summary = {
        "row_count": len(rows),
        "successful_count": len(successful),
        "error_count": len(rows) - len(successful),
        "counts_by_model": dict(Counter(row["model"] for row in rows)),
        "counts_by_condition": dict(Counter(row["condition"] for row in rows)),
        "parse_warning_count": sum(bool(row["parse_warnings"]) for row in successful),
        "note": "Transcript-only is a descriptive baseline and has no single gold reading.",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


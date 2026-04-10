#!/usr/bin/env python3
"""Golden script: survival rate by passenger class + overall survival rate."""

import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Per-class survival rate
    counts = defaultdict(lambda: {"total": 0, "survived": 0})
    total_survived = 0
    for r in rows:
        counts[r["Pclass"]]["total"] += 1
        if r["Survived"] == "1":
            counts[r["Pclass"]]["survived"] += 1
            total_survived += 1

    pclass_survival = {
        k: f"{counts[k]['survived'] / counts[k]['total']:.2f}"
        for k in sorted(counts)
    }

    overall_rate = f"{total_survived / len(rows):.2f}"

    Path("pclass_survival.json").write_text(json.dumps(pclass_survival))
    Path("overall_survival.txt").write_text(overall_rate)
    print(f"Pclass survival: {pclass_survival}")
    print(f"Overall survival rate: {overall_rate}")


if __name__ == "__main__":
    main()

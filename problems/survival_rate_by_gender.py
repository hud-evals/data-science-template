#!/usr/bin/env python3
"""Golden script: compute survival rate by gender."""

import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = defaultdict(lambda: {"total": 0, "survived": 0})
    for r in rows:
        counts[r["Sex"]]["total"] += 1
        if r["Survived"] == "1":
            counts[r["Sex"]]["survived"] += 1

    result = {
        sex: f"{c['survived'] / c['total']:.2f}"
        for sex, c in sorted(counts.items())
    }

    Path("survival_by_gender.json").write_text(json.dumps(result))
    print(f"Survival by gender: {result}")


if __name__ == "__main__":
    main()

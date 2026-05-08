#!/usr/bin/env python3
"""Golden script: survival rate by family size (SibSp + Parch + 1)."""

import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = defaultdict(lambda: {"total": 0, "survived": 0})
    for r in rows:
        family_size = int(r["SibSp"]) + int(r["Parch"]) + 1
        key = "7+" if family_size >= 7 else str(family_size)
        counts[key]["total"] += 1
        if r["Survived"] == "1":
            counts[key]["survived"] += 1

    # Sort: "1" through "6", then "7+"
    sorted_keys = [str(i) for i in range(1, 7)] + ["7+"]
    result = {
        k: f"{counts[k]['survived'] / counts[k]['total']:.2f}"
        for k in sorted_keys
        if k in counts
    }

    Path("family_survival.json").write_text(json.dumps(result))
    print(f"Family survival: {result}")


if __name__ == "__main__":
    main()

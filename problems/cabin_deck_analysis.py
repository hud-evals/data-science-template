#!/usr/bin/env python3
"""Golden script: survival rate by cabin deck letter (first char of Cabin)."""

import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = defaultdict(lambda: {"total": 0, "survived": 0})
    for r in rows:
        if r["Cabin"]:
            deck = r["Cabin"][0]
            counts[deck]["total"] += 1
            if r["Survived"] == "1":
                counts[deck]["survived"] += 1

    result = {
        k: f"{counts[k]['survived'] / counts[k]['total']:.2f}"
        for k in sorted(counts)
    }

    Path("deck_survival.json").write_text(json.dumps(result))
    print(f"Deck survival: {result}")


if __name__ == "__main__":
    main()

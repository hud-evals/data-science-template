#!/usr/bin/env python3
"""Golden script: count passengers in each ticket class."""

import csv
import json
from collections import Counter
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = Counter(r["Pclass"] for r in rows)
    result = {k: counts[k] for k in sorted(counts)}

    Path("class_distribution.json").write_text(json.dumps(result))
    print(f"Class distribution: {result}")


if __name__ == "__main__":
    main()

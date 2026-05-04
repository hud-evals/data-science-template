#!/usr/bin/env python3
"""Golden script: compute mean age of survivors vs non-survivors (excluding missing)."""

import csv
import json
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    survived_ages = [float(r["Age"]) for r in rows if r["Age"] and r["Survived"] == "1"]
    not_survived_ages = [float(r["Age"]) for r in rows if r["Age"] and r["Survived"] == "0"]

    result = {
        "survived_mean": f"{sum(survived_ages) / len(survived_ages):.2f}",
        "not_survived_mean": f"{sum(not_survived_ages) / len(not_survived_ages):.2f}",
    }

    Path("age_stats.json").write_text(json.dumps(result))
    print(f"Age stats: {result}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Golden script: analyze embarkation ports."""

import csv
import json
from collections import Counter
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    embarked_counts = Counter(r["Embarked"] for r in rows if r["Embarked"])
    most_common = embarked_counts.most_common(1)[0][0]

    result = {"most_common": most_common}
    for port in sorted(embarked_counts):
        result[port] = embarked_counts[port]

    Path("embarked_analysis.json").write_text(json.dumps(result))
    print(f"Embarked analysis: {result}")


if __name__ == "__main__":
    main()

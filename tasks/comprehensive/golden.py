#!/usr/bin/env python3
"""Golden script: produce a comprehensive summary with multiple output files."""

import csv
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    survived = sum(1 for r in rows if r["Survived"] == "1")
    ages = [float(r["Age"]) for r in rows if r["Age"]]

    Path("total_passengers.txt").write_text(str(total))
    Path("survival_rate.txt").write_text(f"{survived / total:.2f}")
    Path("mean_age.txt").write_text(f"{sum(ages) / len(ages):.2f}")

    print(f"Total: {total}, Survival rate: {survived / total:.2f}, Mean age: {sum(ages) / len(ages):.2f}")


if __name__ == "__main__":
    main()

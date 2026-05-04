#!/usr/bin/env python3
"""Golden script: compute the average fare paid by all passengers."""

import csv
from pathlib import Path


def main():
    with open("Titanic-Dataset.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_fare = sum(float(r["Fare"]) for r in rows)
    average = total_fare / len(rows)

    Path("average_fare.txt").write_text(f"{average:.2f}")
    print(f"Average fare: {average:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Golden script for the Titanic num_survivors problem.

This script reads the Titanic dataset and counts the number of survivors,
writing the result to num_survivors.txt.

Expected to be run from the workspace directory where Titanic-Dataset.csv exists.
"""

import csv
from pathlib import Path


def count_survivors(csv_path: str) -> int:
    """Count the number of survivors in the Titanic dataset."""
    survivors = 0
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The 'Survived' column: 1 = survived, 0 = did not survive
            if row['Survived'] == '1':
                survivors += 1
    
    return survivors


def main():
    # Look for the dataset in the current directory
    dataset_path = Path('Titanic-Dataset.csv')
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        exit(1)
    
    # Count survivors
    num_survivors = count_survivors(str(dataset_path))
    
    # Write result to output file
    output_path = Path('num_survivors.txt')
    output_path.write_text(str(num_survivors))
    
    print(f"Number of survivors: {num_survivors}")
    print(f"Result written to {output_path}")


if __name__ == '__main__':
    main()

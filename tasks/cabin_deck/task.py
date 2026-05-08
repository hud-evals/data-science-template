from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Extract the deck letter from the Cabin column (first character, e.g. \"C85\" -> \"C\").\n"
        "Exclude passengers with no cabin recorded.\n"
        "Compute survival rate per deck. Round to 2 decimal places.\n\n"
        "Create deck_survival.json mapping deck letter to rate (string), keys sorted.\n"
        'Example: {"A": "0.50", "B": "0.75", ...}'
    ),
    template="titanic_dataset",
    required_outputs={
        "deck_survival.json": '{"A": "0.47", "B": "0.74", "C": "0.59", "D": "0.76", "E": "0.75", "F": "0.62", "G": "0.50", "T": "0.00"}',
    },
)
task.slug = "titanic-cabin-deck"

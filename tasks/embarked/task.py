from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Analyze embarkation ports (Embarked column: C=Cherbourg, Q=Queenstown, S=Southampton).\n"
        "Exclude rows where Embarked is missing.\n"
        "Find the most common port and count passengers from each.\n\n"
        "Create embarked_analysis.json with:\n"
        '- "most_common": the port letter\n'
        "- One key per port letter with its passenger count (sorted alphabetically)\n"
        'Example: {"most_common": "S", "C": 100, "Q": 50, "S": 500}'
    ),
    template="titanic_dataset",
    required_outputs={"embarked_analysis.json": '{"most_common": "S", "C": 168, "Q": 77, "S": 644}'},
)
task.slug = "titanic-embarked-analysis"

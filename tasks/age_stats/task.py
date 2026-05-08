from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Compute the mean age of survivors (Survived=1) vs non-survivors (Survived=0).\n"
        "IMPORTANT: Exclude rows where the Age field is missing/empty.\n"
        "Round each mean to 2 decimal places.\n\n"
        "Create age_stats.json with keys \"survived_mean\" and \"not_survived_mean\",\n"
        "each mapping to the mean as a string.\n"
        'Example: {"survived_mean": "30.00", "not_survived_mean": "31.00"}'
    ),
    template="titanic_dataset",
    required_outputs={"age_stats.json": '{"survived_mean": "28.34", "not_survived_mean": "30.63"}'},
)
task.slug = "titanic-age-stats"

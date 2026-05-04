from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Compute the survival rate for each gender (Sex column).\n"
        "Survival rate = number of survivors (Survived=1) / total for that gender.\n"
        "Round each rate to 2 decimal places.\n\n"
        "Create survival_by_gender.json with a JSON object mapping gender to rate as string.\n"
        "Keys sorted alphabetically.\n"
        'Example: {"female": "0.75", "male": "0.20"}'
    ),
    template="titanic_dataset",
    required_outputs={"survival_by_gender.json": '{"female": "0.74", "male": "0.19"}'},
)
task.slug = "titanic-survival-by-gender"

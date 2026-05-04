from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Compute the average (mean) fare paid by all passengers in the Titanic dataset.\n"
        "Round to 2 decimal places.\n\n"
        "Create a file called average_fare.txt containing only the rounded value."
    ),
    template="titanic_dataset",
    required_outputs={"average_fare.txt": "32.20"},
)
task.slug = "titanic-average-fare"

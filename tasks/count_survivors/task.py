from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Count the number of survivors in the Titanic dataset (Titanic-Dataset.csv).\n"
        "A survivor is a passenger where Survived = 1.\n\n"
        "Create a file called num_survivors.txt containing only the count."
    ),
    template="titanic_dataset",
    required_outputs={"num_survivors.txt": "342"},
)
task.slug = "titanic-count-survivors"

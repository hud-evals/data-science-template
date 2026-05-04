from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Count the number of passengers in each ticket class (Pclass column: 1, 2, 3).\n\n"
        "Create class_distribution.json with a JSON object mapping class number (string key)\n"
        "to passenger count (integer value). Keys sorted: \"1\", \"2\", \"3\".\n"
        'Example format: {"1": 100, "2": 200, "3": 300}'
    ),
    template="titanic_dataset",
    required_outputs={"class_distribution.json": '{"1": 216, "2": 184, "3": 491}'},
)
task.slug = "titanic-class-distribution"

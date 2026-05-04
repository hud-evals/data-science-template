from env import multi_output_analysis

task = multi_output_analysis.task(
    prompt=(
        "Produce a comprehensive summary with three output files:\n\n"
        "1. total_passengers.txt — total number of rows, as an integer.\n"
        "2. survival_rate.txt — overall survival rate, rounded to 2 decimal places.\n"
        "3. mean_age.txt — mean age of passengers with a recorded age\n"
        "   (exclude missing values), rounded to 2 decimal places.\n\n"
        "Each file should contain only the single value."
    ),
    template="titanic_dataset",
    required_outputs={
        "total_passengers.txt": "891",
        "survival_rate.txt": "0.38",
        "mean_age.txt": "29.70",
    },
    output_weights={
        "total_passengers.txt": 0.3,
        "survival_rate.txt": 0.3,
        "mean_age.txt": 0.4,
    },
)
task.slug = "titanic-comprehensive"

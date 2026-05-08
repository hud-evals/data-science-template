from env import multi_output_analysis

task = multi_output_analysis.task(
    prompt=(
        "Perform two analyses:\n\n"
        "1. Survival rate per passenger class (Pclass). Round to 2 decimal places.\n"
        "   Create pclass_survival.json mapping class (string) to rate (string).\n"
        '   Keys sorted: "1", "2", "3".  Example: {"1": "0.65", "2": "0.50", "3": "0.25"}\n\n'
        "2. Overall survival rate across all passengers. Round to 2 decimal places.\n"
        "   Create overall_survival.txt containing only the rate."
    ),
    template="titanic_dataset",
    required_outputs={
        "pclass_survival.json": '{"1": "0.63", "2": "0.47", "3": "0.24"}',
        "overall_survival.txt": "0.38",
    },
)
task.slug = "titanic-correlation-analysis"

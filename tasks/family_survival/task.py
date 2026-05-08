from env import analyze_dataset

task = analyze_dataset.task(
    prompt=(
        "Compute family size per passenger as: SibSp + Parch + 1.\n"
        "Group by family size and compute survival rate per group.\n"
        "Combine sizes >= 7 into a \"7+\" group. Round rates to 2 decimal places.\n\n"
        "Create family_survival.json mapping size (string) to rate (string).\n"
        'Keys ordered: "1","2","3","4","5","6","7+". Only include groups with passengers.\n'
        'Example: {"1": "0.30", "2": "0.50", ...}'
    ),
    template="titanic_dataset",
    required_outputs={
        "family_survival.json": '{"1": "0.30", "2": "0.55", "3": "0.58", "4": "0.72", "5": "0.20", "6": "0.14", "7+": "0.16"}',
    },
)
task.slug = "titanic-family-survival"

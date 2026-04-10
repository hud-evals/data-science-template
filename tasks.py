"""Task definitions for the data-science environment.

Each task is created via scenario.task() and can be run locally or remotely:

    python local_test.py --list
    python local_test.py --task count_survivors --model gpt-4o
"""

from hud_controller.env import analyze_dataset, multi_output_analysis

T = "titanic_dataset"  # template name (shared by all tasks)

# ---------------------------------------------------------------------------
# Easy — binary scoring (analyze_dataset: 1.0 if all outputs match, else 0.0)
# ---------------------------------------------------------------------------

count_survivors = analyze_dataset.task(
    prompt=(
        "Count the number of survivors in the Titanic dataset (Titanic-Dataset.csv).\n"
        "A survivor is a passenger where Survived = 1.\n\n"
        "Create a file called num_survivors.txt containing only the count."
    ),
    template=T,
    required_outputs={"num_survivors.txt": "342"},
)
count_survivors.slug = "titanic-count-survivors"

average_fare = analyze_dataset.task(
    prompt=(
        "Compute the average (mean) fare paid by all passengers in the Titanic dataset.\n"
        "Round to 2 decimal places.\n\n"
        "Create a file called average_fare.txt containing only the rounded value."
    ),
    template=T,
    required_outputs={"average_fare.txt": "32.20"},
)
average_fare.slug = "titanic-average-fare"

class_distribution = analyze_dataset.task(
    prompt=(
        "Count the number of passengers in each ticket class (Pclass column: 1, 2, 3).\n\n"
        "Create class_distribution.json with a JSON object mapping class number (string key)\n"
        "to passenger count (integer value). Keys sorted: \"1\", \"2\", \"3\".\n"
        'Example format: {"1": 100, "2": 200, "3": 300}'
    ),
    template=T,
    required_outputs={"class_distribution.json": '{"1": 216, "2": 184, "3": 491}'},
)
class_distribution.slug = "titanic-class-distribution"

# ---------------------------------------------------------------------------
# Medium — single-output analysis tasks (still binary, but harder problems)
# ---------------------------------------------------------------------------

survival_by_gender = analyze_dataset.task(
    prompt=(
        "Compute the survival rate for each gender (Sex column).\n"
        "Survival rate = number of survivors (Survived=1) / total for that gender.\n"
        "Round each rate to 2 decimal places.\n\n"
        "Create survival_by_gender.json with a JSON object mapping gender to rate as string.\n"
        "Keys sorted alphabetically.\n"
        'Example: {"female": "0.75", "male": "0.20"}'
    ),
    template=T,
    required_outputs={"survival_by_gender.json": '{"female": "0.74", "male": "0.19"}'},
)
survival_by_gender.slug = "titanic-survival-by-gender"

age_stats = analyze_dataset.task(
    prompt=(
        "Compute the mean age of survivors (Survived=1) vs non-survivors (Survived=0).\n"
        "IMPORTANT: Exclude rows where the Age field is missing/empty.\n"
        "Round each mean to 2 decimal places.\n\n"
        "Create age_stats.json with keys \"survived_mean\" and \"not_survived_mean\",\n"
        "each mapping to the mean as a string.\n"
        'Example: {"survived_mean": "30.00", "not_survived_mean": "31.00"}'
    ),
    template=T,
    required_outputs={"age_stats.json": '{"survived_mean": "28.34", "not_survived_mean": "30.63"}'},
)
age_stats.slug = "titanic-age-stats"

embarked = analyze_dataset.task(
    prompt=(
        "Analyze embarkation ports (Embarked column: C=Cherbourg, Q=Queenstown, S=Southampton).\n"
        "Exclude rows where Embarked is missing.\n"
        "Find the most common port and count passengers from each.\n\n"
        "Create embarked_analysis.json with:\n"
        '- "most_common": the port letter\n'
        "- One key per port letter with its passenger count (sorted alphabetically)\n"
        'Example: {"most_common": "S", "C": 100, "Q": 50, "S": 500}'
    ),
    template=T,
    required_outputs={"embarked_analysis.json": '{"most_common": "S", "C": 168, "Q": 77, "S": 644}'},
)
embarked.slug = "titanic-embarked-analysis"

# ---------------------------------------------------------------------------
# Medium — multi-output (partial credit via multi_output_analysis)
# ---------------------------------------------------------------------------

correlation = multi_output_analysis.task(
    prompt=(
        "Perform two analyses:\n\n"
        "1. Survival rate per passenger class (Pclass). Round to 2 decimal places.\n"
        "   Create pclass_survival.json mapping class (string) to rate (string).\n"
        '   Keys sorted: "1", "2", "3".  Example: {"1": "0.65", "2": "0.50", "3": "0.25"}\n\n'
        "2. Overall survival rate across all passengers. Round to 2 decimal places.\n"
        "   Create overall_survival.txt containing only the rate."
    ),
    template=T,
    required_outputs={
        "pclass_survival.json": '{"1": "0.63", "2": "0.47", "3": "0.24"}',
        "overall_survival.txt": "0.38",
    },
)
correlation.slug = "titanic-correlation-analysis"

# ---------------------------------------------------------------------------
# Hard — complex analysis (binary grading, but the problem is harder)
# ---------------------------------------------------------------------------

family_survival = analyze_dataset.task(
    prompt=(
        "Compute family size per passenger as: SibSp + Parch + 1.\n"
        "Group by family size and compute survival rate per group.\n"
        "Combine sizes >= 7 into a \"7+\" group. Round rates to 2 decimal places.\n\n"
        "Create family_survival.json mapping size (string) to rate (string).\n"
        'Keys ordered: "1","2","3","4","5","6","7+". Only include groups with passengers.\n'
        'Example: {"1": "0.30", "2": "0.50", ...}'
    ),
    template=T,
    required_outputs={
        "family_survival.json": '{"1": "0.30", "2": "0.55", "3": "0.58", "4": "0.72", "5": "0.20", "6": "0.14", "7+": "0.16"}',
    },
)
family_survival.slug = "titanic-family-survival"

cabin_deck = analyze_dataset.task(
    prompt=(
        "Extract the deck letter from the Cabin column (first character, e.g. \"C85\" -> \"C\").\n"
        "Exclude passengers with no cabin recorded.\n"
        "Compute survival rate per deck. Round to 2 decimal places.\n\n"
        "Create deck_survival.json mapping deck letter to rate (string), keys sorted.\n"
        'Example: {"A": "0.50", "B": "0.75", ...}'
    ),
    template=T,
    required_outputs={
        "deck_survival.json": '{"A": "0.47", "B": "0.74", "C": "0.59", "D": "0.76", "E": "0.75", "F": "0.62", "G": "0.50", "T": "0.00"}',
    },
)
cabin_deck.slug = "titanic-cabin-deck"

# ---------------------------------------------------------------------------
# Hard — multi-output with weighted SubScores
# ---------------------------------------------------------------------------

comprehensive = multi_output_analysis.task(
    prompt=(
        "Produce a comprehensive summary with three output files:\n\n"
        "1. total_passengers.txt — total number of rows, as an integer.\n"
        "2. survival_rate.txt — overall survival rate, rounded to 2 decimal places.\n"
        "3. mean_age.txt — mean age of passengers with a recorded age\n"
        "   (exclude missing values), rounded to 2 decimal places.\n\n"
        "Each file should contain only the single value."
    ),
    template=T,
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
comprehensive.slug = "titanic-comprehensive"

# ---------------------------------------------------------------------------
# Registry for discovery
# ---------------------------------------------------------------------------

ALL_TASKS = {
    "count_survivors": count_survivors,
    "average_fare": average_fare,
    "class_distribution": class_distribution,
    "survival_by_gender": survival_by_gender,
    "age_stats": age_stats,
    "embarked": embarked,
    "correlation": correlation,
    "family_survival": family_survival,
    "cabin_deck": cabin_deck,
    "comprehensive": comprehensive,
}

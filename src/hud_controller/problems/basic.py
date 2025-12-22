import logging

from hud_controller.spec import PROBLEM_REGISTRY, ProblemSpec

logger = logging.getLogger(__name__)


PROBLEM_REGISTRY.append(
    ProblemSpec(
        id="titanic_dataset_num_survivors",
        template="titanic_dataset",
        golden_script="num_survivors.py",
        description="""
In this problem you will be working with the titanic dataset.

The dataset is available at Titanic-Dataset.csv in the current directory.

Create a file called num_survivors.txt that contains the number of survivors (passengers where Survived = 1).
        
""",
        difficulty="easy",
        required_outputs={"num_survivors.txt": {"exact_text_match": "342"}},
    )
)

PROBLEM_REGISTRY.append(
    ProblemSpec(
        id="titanic_dataset_predict_survival",
        template="titanic_dataset_holdout",
        golden_script="predict_survival.py",
        description="""
In this problem you will be working with the titanic dataset.
Use the Titanic-Dataset-train.csv file as training data, and the Titanic-Dataset-test_x.csv file as test data.
Train a model to predict whether a passenger survived or not.

Save the predictions to a file called predictions.csv with the following columns:
- PassengerId: The passenger ID from the test set
- Survived: The predicted probability of survival (a value between 0 and 1)
""",
        difficulty="easy",
        required_outputs={
            "predictions.csv": {
                "threshold_cross_entropy_loss": {
                    "ground_truth": "/problems/Titanic-Dataset-test_y.csv",
                    "threshold": 0.5,
                }
            }
        },
    )
)

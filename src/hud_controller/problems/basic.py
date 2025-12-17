import logging

from hud_controller.spec import ProblemSpec, PROBLEM_REGISTRY

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
        required_outputs={"num_survivors.txt": "342"},
    )
)

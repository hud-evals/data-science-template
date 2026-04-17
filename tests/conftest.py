"""Shared fixtures for data-science-template test suite."""

import shutil
from pathlib import Path

import pytest

# Repo root is one level up from tests/
REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "problem_templates"
TITANIC_CSV = TEMPLATES_DIR / "titanic_dataset" / "Titanic-Dataset.csv"


@pytest.fixture
def workspace(tmp_path):
    """Provide a temporary workspace with the Titanic dataset copied in."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    shutil.copy2(TITANIC_CSV, ws / "Titanic-Dataset.csv")
    return ws

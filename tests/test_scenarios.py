"""Tests for the data-science environment scenarios.

Follows the hud-blank pattern: exercise scenario generators, helpers, and
grading logic without needing Docker or a running MCP server.
"""

# pyright: reportArgumentType=false

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from env import (
    WORKSPACE,
    _check_outputs,
    _make_prompt,
    _setup_workspace,
    analyze_dataset,
    multi_output_analysis,
)

REPO_ROOT = Path(__file__).parent.parent
TASKS_DIR = REPO_ROOT / "tasks"
TEMPLATES_DIR = REPO_ROOT / "problem_templates"


def run(coro):
    """Helper to run async code in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestMakePrompt:
    """Tests for the _make_prompt helper."""

    def test_contains_description(self):
        prompt = _make_prompt("Count the survivors.")
        assert "Count the survivors." in prompt

    def test_contains_workspace_path(self):
        prompt = _make_prompt("anything")
        assert WORKSPACE in prompt

    def test_mentions_python(self):
        prompt = _make_prompt("anything")
        assert "Python" in prompt or "python" in prompt


class TestCheckOutputs:
    """Tests for the _check_outputs helper."""

    def test_all_correct(self, workspace):
        (workspace / "out.txt").write_text("42\n")
        with patch("env.WORKSPACE", str(workspace)):
            results = _check_outputs({"out.txt": "42"})
        assert results["out.txt"]["passed"] is True

    def test_wrong_value(self, workspace):
        (workspace / "out.txt").write_text("99")
        with patch("env.WORKSPACE", str(workspace)):
            results = _check_outputs({"out.txt": "42"})
        assert results["out.txt"]["passed"] is False
        assert results["out.txt"]["actual"] == "99"

    def test_missing_file(self, workspace):
        with patch("env.WORKSPACE", str(workspace)):
            results = _check_outputs({"missing.txt": "42"})
        assert results["missing.txt"]["passed"] is False
        assert results["missing.txt"]["actual"] is None

    def test_strips_whitespace(self, workspace):
        (workspace / "out.txt").write_text("  42  \n")
        with patch("env.WORKSPACE", str(workspace)):
            results = _check_outputs({"out.txt": "42"})
        assert results["out.txt"]["passed"] is True

    def test_multiple_outputs(self, workspace):
        (workspace / "a.txt").write_text("hello")
        (workspace / "b.txt").write_text("wrong")
        with patch("env.WORKSPACE", str(workspace)):
            results = _check_outputs({"a.txt": "hello", "b.txt": "world"})
        assert results["a.txt"]["passed"] is True
        assert results["b.txt"]["passed"] is False


class TestSetupWorkspace:
    """Tests for the _setup_workspace helper."""

    def test_copies_template(self, workspace):
        with (
            patch("env.WORKSPACE", str(workspace)),
            patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
        ):
            _setup_workspace("titanic_dataset")
        assert (workspace / "Titanic-Dataset.csv").exists()

    def test_clears_existing_files(self, workspace):
        (workspace / "stale.txt").write_text("old data")
        with (
            patch("env.WORKSPACE", str(workspace)),
            patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
        ):
            _setup_workspace("titanic_dataset")
        assert not (workspace / "stale.txt").exists()
        assert (workspace / "Titanic-Dataset.csv").exists()


# ---------------------------------------------------------------------------
# Scenario tests — analyze_dataset
# ---------------------------------------------------------------------------


class TestAnalyzeDatasetScenario:
    """Tests for the analyze_dataset scenario generator."""

    def test_correct_output_scores_1(self, workspace):
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = analyze_dataset(
                    prompt="Count survivors.",
                    template="titanic_dataset",
                    required_outputs={"num_survivors.txt": "342"},
                )
                prompt = await gen.asend(None)
                assert "Count survivors." in prompt

                # Simulate agent writing correct output
                (workspace / "num_survivors.txt").write_text("342")
                reward = await gen.asend("Done")
                assert reward == 1.0

        run(_test())

    def test_wrong_output_scores_0(self, workspace):
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = analyze_dataset(
                    prompt="Count survivors.",
                    template="titanic_dataset",
                    required_outputs={"num_survivors.txt": "342"},
                )
                await gen.asend(None)

                (workspace / "num_survivors.txt").write_text("999")
                reward = await gen.asend("Done")
                assert reward == 0.0

        run(_test())

    def test_missing_output_scores_0(self, workspace):
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = analyze_dataset(
                    prompt="Count survivors.",
                    template="titanic_dataset",
                    required_outputs={"num_survivors.txt": "342"},
                )
                await gen.asend(None)

                # Agent didn't create the file
                reward = await gen.asend("Done")
                assert reward == 0.0

        run(_test())

    def test_partial_outputs_scores_0(self, workspace):
        """Binary grading: all outputs must match for 1.0."""
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = analyze_dataset(
                    prompt="Multiple outputs.",
                    template="titanic_dataset",
                    required_outputs={"a.txt": "correct", "b.txt": "also_correct"},
                )
                await gen.asend(None)

                (workspace / "a.txt").write_text("correct")
                (workspace / "b.txt").write_text("wrong")
                reward = await gen.asend("Done")
                assert reward == 0.0

        run(_test())


# ---------------------------------------------------------------------------
# Scenario tests — multi_output_analysis
# ---------------------------------------------------------------------------


class TestMultiOutputAnalysisScenario:
    """Tests for the multi_output_analysis scenario generator."""

    def test_all_correct_scores_1(self, workspace):
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = multi_output_analysis(
                    prompt="Comprehensive summary.",
                    template="titanic_dataset",
                    required_outputs={
                        "total.txt": "891",
                        "rate.txt": "0.38",
                    },
                )
                prompt = await gen.asend(None)
                assert "Comprehensive summary." in prompt

                (workspace / "total.txt").write_text("891")
                (workspace / "rate.txt").write_text("0.38")
                result = await gen.asend("Done")
                assert result.reward == 1.0

        run(_test())

    def test_none_correct_scores_0(self, workspace):
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = multi_output_analysis(
                    prompt="Summary.",
                    template="titanic_dataset",
                    required_outputs={"a.txt": "1", "b.txt": "2"},
                )
                await gen.asend(None)

                (workspace / "a.txt").write_text("wrong")
                (workspace / "b.txt").write_text("wrong")
                result = await gen.asend("Done")
                assert result.reward == 0.0

        run(_test())

    def test_partial_correct_weighted(self, workspace):
        """One of two outputs correct with equal weights → 0.5."""
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = multi_output_analysis(
                    prompt="Two outputs.",
                    template="titanic_dataset",
                    required_outputs={"a.txt": "yes", "b.txt": "no"},
                )
                await gen.asend(None)

                (workspace / "a.txt").write_text("yes")
                (workspace / "b.txt").write_text("wrong")
                result = await gen.asend("Done")
                assert abs(result.reward - 0.5) < 1e-9

        run(_test())

    def test_custom_weights(self, workspace):
        """Custom weights: only the heavy output is correct."""
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = multi_output_analysis(
                    prompt="Weighted.",
                    template="titanic_dataset",
                    required_outputs={"a.txt": "1", "b.txt": "2"},
                    output_weights={"a.txt": 0.3, "b.txt": 0.7},
                )
                await gen.asend(None)

                (workspace / "a.txt").write_text("wrong")
                (workspace / "b.txt").write_text("2")
                result = await gen.asend("Done")
                assert abs(result.reward - 0.7) < 1e-9

        run(_test())

    def test_subscores_present(self, workspace):
        async def _test():
            with (
                patch("env.WORKSPACE", str(workspace)),
                patch("env.TEMPLATES_DIR", str(TEMPLATES_DIR)),
            ):
                gen = multi_output_analysis(
                    prompt="Check subscores.",
                    template="titanic_dataset",
                    required_outputs={"a.txt": "1", "b.txt": "2"},
                )
                await gen.asend(None)

                (workspace / "a.txt").write_text("1")
                (workspace / "b.txt").write_text("wrong")
                result = await gen.asend("Done")

                assert len(result.subscores) == 2
                names = {s.name for s in result.subscores}
                assert names == {"a.txt", "b.txt"}

        run(_test())


# ---------------------------------------------------------------------------
# Golden script validation — ensure expected outputs in tasks.py are correct
# ---------------------------------------------------------------------------


class TestGoldenScripts:
    """Run golden scripts against the real Titanic dataset and verify
    they produce outputs matching the expected values in each task.py."""

    def _run_golden(self, task_name: str, workspace: Path) -> None:
        """Run tasks/<task_name>/golden.py inside the workspace directory."""
        script = TASKS_DIR / task_name / "golden.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{task_name}/golden.py failed:\n{result.stderr}"

    def test_count_survivors(self, workspace):
        self._run_golden("count_survivors", workspace)
        assert (workspace / "num_survivors.txt").read_text().strip() == "342"

    def test_average_fare(self, workspace):
        self._run_golden("average_fare", workspace)
        assert (workspace / "average_fare.txt").read_text().strip() == "32.20"

    def test_class_distribution(self, workspace):
        self._run_golden("class_distribution", workspace)
        actual = json.loads((workspace / "class_distribution.json").read_text())
        assert actual == {"1": 216, "2": 184, "3": 491}

    def test_survival_by_gender(self, workspace):
        self._run_golden("survival_by_gender", workspace)
        actual = json.loads((workspace / "survival_by_gender.json").read_text())
        assert actual == {"female": "0.74", "male": "0.19"}

    def test_age_stats(self, workspace):
        self._run_golden("age_stats", workspace)
        actual = json.loads((workspace / "age_stats.json").read_text())
        assert actual == {"survived_mean": "28.34", "not_survived_mean": "30.63"}

    def test_embarked(self, workspace):
        self._run_golden("embarked", workspace)
        actual = json.loads((workspace / "embarked_analysis.json").read_text())
        assert actual == {"most_common": "S", "C": 168, "Q": 77, "S": 644}

    def test_correlation(self, workspace):
        self._run_golden("correlation", workspace)
        pclass = json.loads((workspace / "pclass_survival.json").read_text())
        overall = (workspace / "overall_survival.txt").read_text().strip()
        assert pclass == {"1": "0.63", "2": "0.47", "3": "0.24"}
        assert overall == "0.38"

    def test_family_survival(self, workspace):
        self._run_golden("family_survival", workspace)
        actual = json.loads((workspace / "family_survival.json").read_text())
        assert actual == {
            "1": "0.30", "2": "0.55", "3": "0.58", "4": "0.72",
            "5": "0.20", "6": "0.14", "7+": "0.16",
        }

    def test_cabin_deck(self, workspace):
        self._run_golden("cabin_deck", workspace)
        actual = json.loads((workspace / "deck_survival.json").read_text())
        assert actual == {
            "A": "0.47", "B": "0.74", "C": "0.59", "D": "0.76",
            "E": "0.75", "F": "0.62", "G": "0.50", "T": "0.00",
        }

    def test_comprehensive(self, workspace):
        self._run_golden("comprehensive", workspace)
        assert (workspace / "total_passengers.txt").read_text().strip() == "891"
        assert (workspace / "survival_rate.txt").read_text().strip() == "0.38"
        assert (workspace / "mean_age.txt").read_text().strip() == "29.70"

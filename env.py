"""Data Science Environment — Titanic dataset analysis tasks.

Provides:
- Agent tools: bash, editor, shell, apply_patch
- Scenarios representing reusable evaluation patterns:
    analyze_dataset       — single-output analysis with binary grading
    multi_output_analysis — multiple outputs with weighted SubScores

Usage:
    hud dev env:env --stdio          # Run as MCP server
    from env import analyze_dataset  # Import scenarios
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from hud import Environment
from hud.tools.types import EvaluationResult, SubScore

from tools import ApplyPatchTool, BashTool, Command, EditTool, ShellTool

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = "/home/ubuntu/workspace"
TEMPLATES_DIR = "/problem_templates"

_REPO_ROOT = Path(__file__).parent

# Fallback for local development (outside Docker)
if not Path(TEMPLATES_DIR).exists():
    TEMPLATES_DIR = str(_REPO_ROOT / "problem_templates")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env = Environment(name="datascience")

# Tool instances (shared across requests)
_bash_tool = BashTool()
_edit_tool = EditTool()
_shell_tool = ShellTool()
_apply_patch_tool = ApplyPatchTool(base_path=WORKSPACE)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@env.tool()
async def bash(command: str | None = None, restart: bool = False) -> str:
    """Run bash commands. If you need to restart the bash session, set restart to true."""
    result = await _bash_tool(command=command, restart=restart)
    return str(result)


@env.tool()
async def str_replace_based_edit_tool(
    command: str,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
) -> str:
    """Create and edit files using string replacement operations. Use absolute paths."""
    result = await _edit_tool(
        command=cast(Command, command),
        path=path,
        file_text=file_text,
        view_range=view_range,
        old_str=old_str,
        new_str=new_str,
        insert_line=insert_line,
    )
    return str(result)


@env.tool()
async def shell(
    commands: list[str] | None = None,
    timeout_ms: int | None = None,
    max_output_length: int | None = None,
) -> str:
    """Run shell commands with dynamic timeout and output length."""
    result = await _shell_tool(
        commands=commands,
        timeout_ms=timeout_ms,
        max_output_length=max_output_length,
    )
    return str(result)


@env.tool()
async def apply_patch(
    type: str | None = None,
    path: str | None = None,
    diff: str | None = None,
) -> str:
    """Create, update, or delete files using structured V4A diffs."""
    result = await _apply_patch_tool(type=type, path=path, diff=diff)
    return str(result)


@env.tool()
async def hud_validate() -> str:
    """Run the test suite to validate the environment is working correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output or f"pytest exited with code {result.returncode}")
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_workspace(template: str) -> None:
    """Copy a problem template into the workspace directory."""
    template_path = Path(TEMPLATES_DIR) / template

    os.makedirs(WORKSPACE, exist_ok=True)
    # Clear existing workspace contents
    for item in Path(WORKSPACE).iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Copy template files into workspace
    for item in template_path.iterdir():
        dst = Path(WORKSPACE) / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    # Ensure ubuntu owns workspace (if running as root)
    if os.getuid() == 0:
        subprocess.run(["chown", "-R", "ubuntu:ubuntu", WORKSPACE], check=False)

    logger.info("Workspace set up with template '%s'", template)


def _make_prompt(description: str) -> str:
    """Format a task description into a full agent prompt."""
    return f"""You will be working on a data science task.
The workspace has been set up at {WORKSPACE} with the necessary data files.

Python3 is available. You can use standard library modules (csv, json, etc.).
If you need additional packages, you can install them with pip.

Use the tools provided to complete the following task:

{description}"""


def _outputs_match(output_file: str, actual: str, expected: str) -> bool:
    """Compare actual vs expected. For .json files, compare parsed structures
    so pretty-printing and key order don't break grading."""
    if output_file.endswith(".json"):
        try:
            return json.loads(actual) == json.loads(expected)
        except json.JSONDecodeError:
            return False
    return actual == expected


def _check_outputs(required_outputs: dict[str, str]) -> dict[str, dict]:
    """Check each required output file against its expected value.

    Returns {filename: {passed, expected, actual}}.
    """
    results = {}
    for output_file, expected in required_outputs.items():
        file_path = Path(WORKSPACE) / output_file
        entry: dict[str, Any] = {"passed": False, "expected": expected.strip()}
        if not file_path.exists():
            entry["actual"] = None
        else:
            actual = file_path.read_text().strip()
            entry["actual"] = actual
            entry["passed"] = _outputs_match(output_file, actual, expected.strip())
        results[output_file] = entry
    return results


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@env.scenario("analyze_dataset", exclude_tools=["hud_validate"])
async def analyze_dataset(
    prompt: str,
    template: str,
    required_outputs: dict[str, str],
):
    """Analyze a dataset and produce one or more output files.

    Binary grading: 1.0 if every required output matches, 0.0 otherwise.
    Good for straightforward analysis tasks with a single correct answer.
    """
    _setup_workspace(template)
    yield _make_prompt(prompt)

    results = _check_outputs(required_outputs)
    all_passed = all(r["passed"] for r in results.values())
    yield 1.0 if all_passed else 0.0


@env.scenario("multi_output_analysis", exclude_tools=["hud_validate"])
async def multi_output_analysis(
    prompt: str,
    template: str,
    required_outputs: dict[str, str],
    output_weights: dict[str, float] | None = None,
):
    """Analyze a dataset and produce multiple output files with weighted scoring.

    Each output is scored independently (1.0 correct, 0.0 wrong).
    The final reward is the weighted sum across outputs, reported as SubScores.
    If output_weights is omitted, all outputs are weighted equally.
    """
    _setup_workspace(template)
    yield _make_prompt(prompt)

    results = _check_outputs(required_outputs)
    outputs = list(required_outputs.keys())

    # Build normalised weights
    weights = dict(output_weights or {})
    if not weights:
        w = 1.0 / len(outputs)
        weights = {name: w for name in outputs}
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    subscores = [
        SubScore(
            name=name,
            weight=weights.get(name, 1.0 / len(outputs)),
            value=1.0 if results[name]["passed"] else 0.0,
        )
        for name in outputs
    ]
    reward = sum(s.weight * s.value for s in subscores)

    yield EvaluationResult(
        reward=reward,
        done=True,
        content=f"{sum(1 for s in subscores if s.value > 0)}/{len(subscores)} outputs correct",
        subscores=subscores,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env.run(transport="stdio")

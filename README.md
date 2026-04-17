# Data Science Environment

A data science coding environment where agents analyze datasets in a sandboxed workspace and are graded by comparing output files against expected values.

## Setup

```bash
uv sync
cp .env.example .env                # Injected into the deployed container for grading
hud set HUD_API_KEY=your-key-here   # CLI auth, get one at hud.ai/project/api-keys
```

## Deploy & Run

```bash
hud deploy .                              # deploy the environment (once)
hud sync tasks <taskset-name>             # push tasks to a taskset (fast, re-run on every task change)
hud eval <taskset-name> --remote --full
```

**Iteration loop:** `hud deploy` is the slow step — run it once. After that, edit `tasks.py` and re-run `hud sync tasks` (takes seconds). Only redeploy when `env.py` or the Dockerfile changes.

See [Deploy & Go Remote](https://docs.hud.ai/building/running-at-scale) for deploy flags, secrets, and auto-deploy options.

## Tasks

10 tasks using the Titanic dataset across three difficulty levels:

**Easy** — count survivors, compute average fare, class distribution

**Medium** — survival rate by gender, age statistics, embarkation analysis, correlation analysis

**Hard** — family size survival, cabin deck analysis, comprehensive multi-output summary

Agents get a description of the analysis to perform, write Python in a sandboxed workspace, and are graded by exact match of output files against expected values.

## Documentation

To learn more about tasks, evaluations, and running at scale see the [full docs](https://docs.hud.ai).

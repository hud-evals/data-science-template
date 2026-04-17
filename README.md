# Data Science Environment

A data science coding environment where agents analyze datasets in a sandboxed workspace and are graded by comparing output files against expected values.

## Quick Start

```bash
uv sync                # install dependencies
hud deploy .           # build and deploy to HUD platform
hud sync tasks <name>  # upload task definitions
```

## Tasks

10 tasks using the Titanic dataset across three difficulty levels:

**Easy** — count survivors, compute average fare, class distribution

**Medium** — survival rate by gender, age statistics, embarkation analysis, correlation analysis

**Hard** — family size survival, cabin deck analysis, comprehensive multi-output summary

Agents get a description of the analysis to perform, write Python in a sandboxed workspace, and are graded by exact match of output files against expected values.

## Documentation

To learn more about tasks, evaluations, and running at scale see the [full docs](https://docs.hud.ai).

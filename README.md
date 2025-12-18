# Data Science Agent Evaluation Framework

## Overview

This is a framework for creating and evaluating AI agent tasks focused on data science problems. It provides a structured approach to:
- Define data science tasks with clear specifications
- Grade agent solutions by comparing output files to expected values
- Manage multiple task difficulties (easy, medium, hard)
- Run tasks in isolated Docker environments with proper grading

## Project Structure

```
.
├── src/hud_controller/          # Main framework code
│   ├── app.py                   # Main MCP server and entry points
│   ├── spec.py                  # Core specifications (ProblemSpec, Grade)
│   ├── grading_runner.py        # Output validation and grading logic
│   ├── utils.py                 # Utility functions
│   ├── setup.py                 # Environment setup
│   ├── problems/                # Task definitions
│   │   └── basic.py             # Problem registrations
│   └── tools/                   # MCP tools for agent interaction
│       ├── base.py              # Base tool definitions
│       ├── bash.py              # Bash execution
│       ├── edit.py              # File editing
│       ├── shell.py             # Shell commands
│       └── apply_patch.py       # Patch application
├── problem_templates/           # Data files for each problem template
│   └── titanic_dataset/         # Example: Titanic dataset template
│       └── Titanic-Dataset.csv
├── problems/                    # Golden scripts that produce expected outputs
│   └── num_survivors.py         # Example: counts Titanic survivors
├── utils/
│   └── imagectl3.py             # Docker image build/push/validate tool
├── pyproject.toml               # Python package configuration
├── Dockerfile                   # Container setup
└── README.md                    # This file
```

## Core Concepts

### 1. Problem Definition

Problems are defined using the `ProblemSpec` data class:

```python
ProblemSpec(
    id="titanic_dataset_num_survivors",  # Unique problem ID
    template="titanic_dataset",           # Folder name in problem_templates/
    golden_script="num_survivors.py",     # Script in problems/ that produces correct output
    description="""
In this problem you will be working with the titanic dataset.

The dataset is available at Titanic-Dataset.csv in the current directory.

Create a file called num_survivors.txt that contains the number of survivors.
    """,
    difficulty="easy",
    required_outputs={"num_survivors.txt": "342"},  # Expected outputs (trimmed)
)
```

### 2. Templates and Golden Scripts

**Templates** (`problem_templates/`):
- Contain data files and any starter code
- Copied to `/home/ubuntu/workspace` when the Docker image is built
- Each template is a folder (e.g., `titanic_dataset/`)

**Golden Scripts** (`problems/`):
- Python scripts that solve the problem correctly
- Used during validation to verify expected outputs are correct
- Run from the workspace directory during grading

### 3. Output-Based Validation

Tasks are graded by:
1. Copying the workspace to a clean grading directory
2. Copying and running the golden script
3. Comparing each required output file against expected values (string match after trimming)

## Creating New Tasks

### Step 1: Create the Template

Create a new folder in `problem_templates/` with your data files:

```
problem_templates/
└── my_dataset/
    ├── data.csv
    └── config.json
```

### Step 2: Write the Golden Script

Create a Python script in `problems/` that produces the expected outputs:

```python
# problems/my_solution.py
#!/usr/bin/env python3
"""Golden script for my_dataset problem."""

import csv
from pathlib import Path

def main():
    # Read data from the workspace (cwd)
    with open('data.csv') as f:
        data = list(csv.DictReader(f))
    
    # Compute the answer
    result = len(data)
    
    # Write the output file
    Path('answer.txt').write_text(str(result))

if __name__ == '__main__':
    main()
```

### Step 3: Register the Problem

Add the problem to `src/hud_controller/problems/basic.py` (or create a new file):

```python
from hud_controller.spec import ProblemSpec, PROBLEM_REGISTRY

PROBLEM_REGISTRY.append(
    ProblemSpec(
        id="my_dataset_count",
        template="my_dataset",
        golden_script="my_solution.py",
        description="""
Your task description here. Explain what the agent should do.

The data is available in data.csv.
Create a file called answer.txt with the number of rows.
        """,
        difficulty="easy",
        required_outputs={"answer.txt": "100"},  # Expected value after trim
    )
)
```

### Step 4: Validate Your Problem

Use `imagectl3.py` to build and validate:

```bash
# Build and validate a specific problem
uv run utils/imagectl3.py myprefix_ -bv --ids my_dataset_count

# Build and validate all problems
uv run utils/imagectl3.py myprefix_ -bv
```

The validation workflow:
1. Copies the template to a temp directory
2. Runs the golden script
3. Verifies all required outputs match expected values

## Running Tasks

### Setup Environment

```bash
uv sync
```

### Build, Validate, and Generate JSON

```bash
# Build all images with prefix, validate, and generate JSON configs
uv run utils/imagectl3.py datascience_ -bvj

# Run with parallel jobs for faster builds
uv run utils/imagectl3.py datascience_ -bvj --jobs 4
```

### Run HUD Eval Locally

```bash
uv run hud local-claude-hud.json claude --max-steps 50
# or for OpenAI
uv run hud local-openai-hud.json openai --max-steps 50
```

### Run HUD Eval Remotely

Push images to a registry first:

```bash
# Build, validate, generate JSON, and push
uv run utils/imagectl3.py yourusername/datascience_ -bvjp --jobs 4
```

Then run remotely:

```bash
uv run hud remote-claude-hud.json claude --max-steps 50
```

## Configuration

### Environment Variables

Key environment variables:

- `MCP_TESTING_MODE` - Enable testing tools (default: "1")
- `HINTS` - Hint mode: "none" or "all" (default: "none")
- `PROBLEM_ID` - The problem ID to run
- `TEMPLATE` - The template folder to use

### Docker Build Args

The Dockerfile accepts these build arguments:

- `TEMPLATE` - Which template folder to copy to `/home/ubuntu/workspace`
- `PROBLEM_ID` - The problem ID for the image
- `HINTS` - Whether to include hints in the prompt

### Hints

You can add hints to problems:

```python
from hud_controller.spec import ProblemSpec, HintSpec, PROBLEM_REGISTRY

PROBLEM_REGISTRY.append(
    ProblemSpec(
        id="my_problem",
        # ... other fields ...
        hints=[
            HintSpec(
                hint_type="legit",
                text="The Survived column contains 0 or 1",
                why_legitmate="This is documented in the dataset description"
            ),
        ],
    )
)
```

Build with hints enabled:

```bash
uv run utils/imagectl3.py prefix_ -bv --hints all
```

## Best Practices

### Task Design

1. **Clear Descriptions**: Provide detailed, unambiguous task descriptions
2. **Focused Scope**: Each task should test one concept or skill
3. **Realistic Scenarios**: Base tasks on real-world data science problems
4. **Fair Hints**: If providing hints, ensure they guide without giving away the solution

### Golden Script Design

1. **Minimal Dependencies**: Use Python standard library when possible
2. **Clear Output**: Write exactly what's expected, nothing more
3. **Error Handling**: Handle missing files gracefully for better error messages
4. **Comments**: Document what the script does for maintainability

### Template Design

1. **Self-Contained**: Include all necessary data files
2. **Reasonable Size**: Keep datasets small enough for quick Docker builds
3. **Clear Naming**: Use descriptive file names
4. **No Secrets**: Don't include expected outputs in the template

### Output Validation

1. **Trimmed Comparison**: Values are compared after stripping whitespace
2. **Exact Match**: The output must exactly match the expected value
3. **Multiple Outputs**: You can require multiple output files
4. **Simple Values**: Keep expected outputs simple (numbers, short strings)

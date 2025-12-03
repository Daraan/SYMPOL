# Ray Utilities Development Guide

## Project Purpose

This project works with JAX and RLlib to provide a framework for training reinforcement learning agents using decision tree-based policies.
When working with JAX keep in mind that it works a bit differently than standard Python code due to its functional programming paradigm, immutability, and just-in-time compilation.
When you work on JAX related code always consider if the code needs to be JIT-compiled or if it can run in standard Python mode for easier debugging.
If it is not yet jittable consider refactoring it to be jittable if possible.
We only care for jittable code in the models and learner.

## Core Architecture

### The Setup → Trainable → Tuner Pipeline

1. **Setup Phase** (`ExperimentSetupBase`): Parse CLI args → build `AlgorithmConfig` → freeze config
2. **Trainable Creation** (`DefaultTrainable`): Wrap frozen config in trainable class for distributed execution
3. **Execution** (`run_tune()`): Create `Tuner` → execute trials → upload offline logs

### Why This Matters for Development

- **Adding new args:** Must understand which parser class to extend and which annotation (`AlwaysRestore`, `NeverRestore`) controls checkpoint behavior
- **Creating callbacks:** Choose `RLlibCallback` (runs in training loop) vs Tune `Callback` (runs in Tuner, sees all trials)
- **Checkpoint restoration:** Setup, args, and config_files are all checkpointed together - changes must be backward compatible


## Project Structure & Key Files

### Where to Add Code

| Adding... | Go to... | Notes |
|-----------|----------|-------|
| New CLI argument | `sympol/rllib_port/extended_args.py` | Extend appropriate parser class, use `AlwaysRestore`/`NeverRestore` annotations |
| In general | `sympol/rllib_port/sympol` and `sympol/rllib_port/core/*` | For jax related and interface code |
| In general | `sympol/rllib_port/` | All Ray RLlib-specific code lives here |

### Critical Files (Read These First)

**Core Flow:**
1. `config/parser/default_argument_parser.py` (650 lines) - All CLI args, inheritance chain with annotations
2. `setup/experiment_base.py` (1500+ lines) - Setup lifecycle: parse → config → trainable → tuner
3. `training/default_class.py` (1500+ lines) - Trainable implementation: algorithm creation, checkpointing, progress tracking

**Key Helpers:**
- `training/helpers.py` - Checkpoint restoration, config copying, metric processing
- `constants.py` - Metric keys, result keys, project-wide constants
- `misc.py` - Utility functions used across codebase

## Coding Conventions

### Mandatory Rules

**Logging (Strictly Enforced):**
- No print() statements: Always use logger when using python. Print statements don't respect log levels and aren't captured properly in distributed Ray workers.

```python
logger = logging.getLogger(__name__)
logger.info("Step %s: loss=%s", step, loss)  # ✓ REQUIRED: %s formatting
logger.info(f"Step {step}: loss={loss}")      # ✗ FORBIDDEN: f-strings break lazy evaluation
```

**Type Hints:** Required on all public functions/methods. Use `from __future__ import annotations` for forward references.


**Editing Rules**
- Only edit files under `ray_utilities/` (and project `docs/` and `tests/`); never touch site-packages or env directories
- Place new code in the correct folder (see sections: Quick Reference: When Working On..., and Project Structure & Key Files)
- Prefer minimal diffs; keep changes localized
- Do not move code unless necessary or instructed
- Keep public APIs stable; update tests and pyproject.toml extras if changed
- Use type annotations for all public functions and methods
- Prefer `pathlib.Path` over `os.path` for new code
- do not remove explaining comments unless they are incorrect or obsolete, also never remove comments starting with with (noqa, pyright, pragma, fmt, ruff, isort), expect when they are incorrect, obsolete or may shadow problematic code.

### Testing Workflow

Tests are stored in the `test/` directory. Use `pytest` with verbose output for running tests. Example:

```bash
pytest test/test_trainable.py -v
```

**Test Utilities** (`testing_utils.py`): Use `TestHelper`, `DisableLoggers`, `InitRay`, `patch_args` for test isolation. Example:

```python
from ray_utilities.testing_utils import DisableLoggers, InitRay, TestHelper, patch_args

class MyTest(DisableLoggers, InitRay, TestHelper, num_cpus=4):
    @patch_args("--iterations", 1)
    def test_something(self):
        # Test code here
        pass
```

### Entry Point Pattern

When opening a shell or terminal always use `../env/bin/activate` first to activate the virtual environment.

```python
# experiments/default_training.py pattern
if __name__ == "__main__":
    ray.init(runtime_env=runtime_env)
    with DefaultArgumentParser.patch_args("--seed", "42", "--wandb", "offline+upload"):
        with PPOSetup(config_files=["experiments/default.cfg"]) as setup:
            setup.config.training(num_epochs=10)  # Adjust config in with-block
        results = run_tune(setup)  # Executes the experiment
```

**Why `with setup:`?** Config changes inside the block are tracked for checkpoint reloads. After `__exit__`, config is frozen and trainable is created.


## Common Pitfalls

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Modifying frozen config | `AttributeError: Cannot set attribute ... already frozen` | Use `with setup:` or `setup.unset_trainable()` |
| Using `print()` | Output not in logs, breaks distributed execution | Use `logger = logging.getLogger(__name__)` |
| F-strings in logging | Performance issues, breaks filtering | Use `logger.info("msg: %s", var)` |
| Bare `except:` | Catches KeyboardInterrupt, masks bugs | Catch specific exceptions |
| Forgetting `with setup:` | Config not frozen, trainable not created | Always use context manager or call `create_trainable()` |
| global variable usage on remote | (silently) wrong configuration chosen | Avoid modifying globals that are used by remote classes (`RLlibCallback`, `ray.remote`, `DefaultTrainable`). Tuner `Callback` and `ExperimentSetupBase` classes are fine. | Pass via config or pass with `ray.put/get` or `ray.remote` |
| Online WandB/Comet in tests | Tests hang, is slow | mock to avoid network I/O

- **Don't modify config after `with setup:` block** - changes won't be tracked for checkpoint restoration
- **Don't call `run_in_terminal` for Python execution** - use `mcp_pylance_mcp_s_pylanceRunCodeSnippet` for cleaner output


# Testing and Agent behavior

When you open a console always run "source ../env/bin/activate" first to activate the virtual environment.
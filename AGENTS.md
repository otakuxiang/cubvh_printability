# Repository Guidelines

## Project Structure & Module Organization

This repository uses a `src` layout. Python package code lives in `src/cubvh_printability/`; the public client and configuration are in `client.py`, the CLI is in `cli.py`, and printability implementation modules are alongside them. OptiX/CUDA helper sources and headers are packaged under `src/cubvh_printability/optix_src/`. Tests are in `tests/` (currently `tests/test_client.py`). `build/` and `src/*.egg-info/` are generated artifacts; do not edit them by hand or include them in focused changes.

## Build, Test, and Development Commands

Install the package and its editable dependencies with:

```bash
python -m pip install -e .
```

Run the test suite with:

```bash
python -m pytest
```

The tests exercise configuration validation, lazy backend loading, and missing-input handling without requiring CUDA. For a local CLI check after installing, use `cubvh-printability input.glb --min-thickness 0.05 --size 50 --output repaired.ply`. Runtime printability operations require `cubvh` and its CUDA/OptiX dependencies.

## Coding Style & Naming Conventions

Target Python 3.10+ and use four spaces, type hints, concise docstrings, and standard-library imports before third-party imports. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase names for constants. Preserve the client’s lazy import behavior so importing the package does not initialize CUDA. Keep public API changes synchronized across `client.py`, `__init__.py`, and the CLI when applicable.

## Testing Guidelines

Use pytest and name tests `test_<behavior>`. Add focused regression tests under `tests/` for validation, API delegation, and CLI-facing behavior. Keep tests deterministic and avoid requiring a GPU when a backend stub can verify the same contract. No coverage threshold or formatter configuration is currently declared.

## Commit & Pull Request Guidelines

Git history is not present in this checkout, so no existing commit convention can be verified. Use short, imperative commit subjects (for example, `Add output path validation`) and keep each commit focused. Pull requests should explain the user-visible behavior, list tests run, note CUDA/OptiX assumptions, and include CLI output or screenshots when changing user-facing behavior.

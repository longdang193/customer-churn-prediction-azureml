# Dependencies

Centralized project dependencies and workflow for reproducible environments.

Centralized project dependencies and workflow for reproducible environments.

## Files

- `requirements.in` — application/runtime dependencies (unpinned)
- `dev-requirements.in` — developer tooling (unpinned)
- `requirements.txt` — pinned, compiled from `requirements.in` (generated)
- `dev-requirements.txt` — pinned, compiled from `dev-requirements.in` (generated)

## Manage dependencies with pip-tools

Install pip-tools:

```bash
python -m pip install --upgrade pip
python -m pip install pip-tools
```

Pin runtime and dev requirements:

```bash
pip-compile requirements.in -o requirements.txt
pip-compile dev-requirements.in -o dev-requirements.txt
```

Install:

```bash
python -m pip install -r requirements.txt
python -m pip install -r dev-requirements.txt
```

## Important Notes

- All packages listed in `requirements.in` are runtime dependencies required for the project.
- If you change `requirements.in` or `dev-requirements.in`, re-run `pip-compile` and commit the updated `*.txt` files.


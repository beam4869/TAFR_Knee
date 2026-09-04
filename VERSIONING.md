# Versioning and releases

TAFR-Knee uses Semantic Versioning: `MAJOR.MINOR.PATCH`.

- `PATCH` releases fix implementation or documentation defects without changing the
  intended public behavior.
- `MINOR` releases add backward-compatible APIs, adapters, scalarizations, metrics,
  or benchmark support.
- `MAJOR` releases may change public signatures, result semantics, or default
  certification behavior.

The version appears in both `pyproject.toml` and `src/tafrknee/_version.py`. A release
must update both values, move entries from `Unreleased` into a dated changelog
section, pass the complete validation suite, and receive an annotated Git tag named
`vMAJOR.MINOR.PATCH`.

Release checklist:

1. Run `python -m pytest`.
2. Run `python -m ruff check src tests examples`.
3. Run `python -m build` and inspect both wheel and source archive.
4. Verify installation and import from the built wheel in a clean environment.
5. Commit the release metadata with `Release vMAJOR.MINOR.PATCH`.
6. Create and push an annotated tag.


# Public artifact release, 6 September 2026

The repository owner explicitly approved public upload of the raw experiment
results, figures and compiled experiment Note PDF. This resolves the earlier
automatic-review block. The experiment branch now includes the historical pilot
and the completed continuation artifacts, with byte-level integrity metadata.

## Available files

- [Continuation summaries and figures](results/continuation/)
- [Historical pilot figures](results/figures/) and [tables](results/tables/)
- [Compiled historical experiment Note](note/TAFR_Experiment_Note.pdf)
- [Raw and processed result archives](results/artifacts/)
- [Complete SHA-256 manifest](provenance/public_artifacts_2026-09-06.json)

The manifest covers 3,080 original files (204,800,577 uncompressed bytes).
Eight ZIP archives preserve raw/processed measurements, per-run checkpoints and
the two largest continuation record tables. Figures and smaller summaries are
directly browsable. Archives preserve original file paths and contents; no
measurements are dropped or rounded during packaging.

From the repository root, verify and restore the files with standard Python:

```bash
python experiments/restore_artifacts.py --verify-only
python experiments/restore_artifacts.py
python -m experiments.runners.report_continuation
```

The restore tool checks archive hashes and every member before extraction. It
preserves identical existing files and stops if another version would be replaced.
Use `--destination /path/to/separate-directory` for a separate extraction, or
explicitly use `--overwrite` when replacing existing measurements is intended.

The original pilot Note is historical and has not been revised to describe the
continuation. Current counts and limitations are in
[CONTINUATION.md](CONTINUATION.md) and
[completion.json](results/continuation/completion.json). Source papers,
the proposal/CV and local Git-history bundles are not included.

## Versions and scientific scope

The continuation's final analysis code is commit
`7e379592f3d920eb59172d3dc67b1fec3f2d41c7`; execution-source mappings are retained in
[continuation_commit_map.json](provenance/continuation_commit_map.json).
The core package remains unchanged from `9f7a65d7` (v0.1.0), and external
dependencies remain pinned. Publishing artifacts does not complete the remaining
final-paper experiments or strengthen the scientific claims.

The separately delivered earlier `TAFR_Experiment_Release.zip` is a historical
snapshot with SHA-256
`a09274629b65d3497fc4841cacd9cab1451c6f4368525a337e4b710fd79bc593`.
This public release supersedes its artifact-availability limitation; it does not
redistribute that archive's local execution-history bundle.

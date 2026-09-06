# Public release status

The 6 September continuation adds tested experiment code and reporting tools;
see [CONTINUATION.md](CONTINUATION.md). GitHub CLI transport can read this public
repository but has no write credential in the execution environment. Code is
published through the authenticated GitHub connector, with commit mappings in
`provenance/continuation_commit_map.json`. A new attempt to publish the compiled
experiment PDF was rejected by automatic approval review because the review
required explicit authorization to publicly disclose that unpublished artifact.
The PDF and full scientific result archive therefore remain unpublished here.

This branch contains the current experiment code, configurations, measured summary
tables, plotting scripts, and reviewed LaTeX note source. The original core package
remains unchanged from `9f7a65d7`. Public external dependencies retain pinned commits.

Automatic approval review blocked publication of the large raw/processed result
bundles and generated binary artifacts. This branch therefore does not contain the
complete pilot result archive. The earlier raw checkpoints already present are
historical diagnostics; they do not replace the corrected v2 SNEE data summarized
in the final tables. No missing or blocked output was fabricated.

The separately delivered `TAFR_Experiment_Release.zip` contains the full original
measurements, all figures, the compiled note, and exact local execution history.
Archive SHA-256:
`a09274629b65d3497fc4841cacd9cab1451c6f4368525a337e4b710fd79bc593`.
Restore that release's `experiments/results/` and note PDF before rebuilding the
report from cached measurements, or regenerate results with the supplied runners.
The Note source package separately includes every PDF figure needed for compilation.

The complete 30-seed final benchmark matrix remains pending scientific gates
documented in `experiments/STATUS.md`; this is a mechanism-validation and pilot release.

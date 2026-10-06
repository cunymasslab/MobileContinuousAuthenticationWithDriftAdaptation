# Reanalysis verification records — 3–4 October 2026

This directory preserves the maintainer's execution and comparison records from Compute B. It is an evidence supplement, not a claim that the entire replication package is complete. The executions occurred on Compute B; packaging inspected the uploaded files and compared saved bytes without rerunning the research analyses.

## Recorded outcomes

| Component | Outcome supported by the retained evidence | Boundary |
|---|---|---|
| M7–M8 statistical reanalysis | Execution returned 0. The maintainer's comparison records exact equality for all four statistical CSVs and equality of the text report after normalizing only its candidate-input path. | The CSV bytes from that new run are not in this evidence archive; its comparison log records their hashes and matches. The accepted CSVs are distributed separately under `artifacts/reference/m7_repaired_20260901/`. |
| Matched-stream report | Execution returned 0. Direct comparison of the two reports in this archive confirms that only the order of the four initial coverage-count lines differs. Every byte outside that block is identical. | Strict byte comparison is **false** and its original log and diff are preserved. Agreement is at the precision printed in the reports, not a full-precision numerical-output or figure comparison. |
| M9 configuration-weighted sweep | Execution returned 0. Both comparison CSVs are present and are byte-for-byte equal; their SHA-256 is `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8`. | This is the sweep-comparison table only, not the separate M9 user-level confidence interval, paired test, or figure reproduction. |

### M7–M8 record

The recorded run used Python 3.13.2, NumPy 2.2.6, Pandas 2.2.3, and SciPy 1.15.3 in the maintainer's existing environment. It used the accepted summary SHA-256 `7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098` and script SHA-256 `c9d6f4b1364748a2fc80d9c11894a6e3f654cbf0d40bad1cf5f2d0ebddb11fc7`. The run retained 9,216 complete M6/M7/M8 configurations across 89 target users from 9,558 input rows with zero duplicate keys. The source was merged in PR #4; the new-run comparison record is [here](records/m7/output_comparison__1y0flmt.txt) and the execution record is [here](records/m7/execution.log).

This was not a fresh-install M7 check. Do not transfer the matched-stream environment's clean-install status to the earlier M7 run merely because the three scientific-library versions agree.

### Matched-stream ordering exception

The saved reference orders the four coverage lines as `cross-activity`, `cross-both`, `temporal`, `cross-posture`. The reproduced report orders them as `cross-both`, `cross-activity`, `cross-posture`, `temporal`. The first two labels each have 3,132 rows; the latter two each have 1,566. This is the only difference in the complete saved report bytes. The recorded report hashes are:

- Reference: `90fb8051bbba1d027e2ec289553c28f8640dc5c7c2d5f0997a447bb967028295`.
- Reproduced: `b492ade6606ace40f0aa890a25de3426a0f007d2509db45aea034afd197a5104`.

The [strict comparison](records/matched_stream/report_comparison.txt), [exact diff](records/matched_stream/report_difference.diff), [reference report](records/matched_stream/reference/analysis_confound.txt), and [reproduced report](records/matched_stream/results_confound/analysis_confound.txt) are retained unchanged. Neither report has been sorted, rewritten, rounded, or substituted for the other. The [execution record](records/matched_stream/execution.log) records the five input/source hashes, including the accepted repaired factorial summary copied under `results_stage4_merged_summary.csv` for the script's M1 baseline check.

The mixed-effects boundary and non-positive-definite Hessian warnings appear in both reports. They are preserved, not resolved by successful reproduction. Deprecation and dtype warnings in the execution record are also retained. Reproducing a historical report is not an endorsement of every explanatory statement in that report or a new assessment of its statistical assumptions.

### M9 comparison code

[analysis/reproduce_m9_comparison.py](../../../analysis/reproduce_m9_comparison.py) is a byte-for-byte copy of the reporting-only script executed on Compute B; its SHA-256 is `50aadce3c5450182fa2c3e64371e0d1bb364b1cd13eb8fb941ed2edff7f5762a`. The original two function bodies (`output_paths` and `write_comparison`) were extracted from `run_m9_lr_sweep.py` without changing their abstract syntax. The training driver's imports and startup are not included. The original driver snapshot is already preserved under `artifacts/reference/matched_stream_m9/source_snapshot/`.

This tested script uses fixed paths relative to its working directory. It requires `validation_users.txt` and five summaries below `results_m9_lr_sweep/` (M2 at 0.002 and M9 at 0.0001, 0.0005, 0.001, 0.002). It writes `results_m9_lr_sweep/comparison.csv` and will overwrite an existing file at that path. It also warns and skips a missing summary. Therefore it is **not** a safe in-place command against the reference directory and does not by itself establish complete input coverage. The recorded check staged all five inputs in a new directory, kept reference `comparison.csv` separately, and ran:

```text
<reanalysis-python> -I -B -u reproduce_m9_comparison.py
```

See the [execution record](records/m9_sweep/execution.log) and [exact output comparison](records/m9_sweep/output_comparison.txt). No model was trained.

## Reproduction environment

The [requested requirements](records/environment/requirements-requested.txt) specify six packages. The supplied installation report and resolved requirements describe 17 installed packages in a separate CPython 3.13.2 Linux x86_64 environment. The [dependency check](records/environment/dependency-check.log) reports no broken requirements and the [import check](records/environment/import-check.log) reports successful imports of NumPy, Pandas, SciPy, Matplotlib, Statsmodels, Patsy, and the Statsmodels formula interface tested in the command.

[requirements/reanalysis-python313.txt](../../../requirements/reanalysis-python313.txt) is an unchanged copy of the resolved version list, including transitive dependencies. It is a **version snapshot of the tested reanalysis environment**, not the original model-training environment and not a wheel-hash-locked or cross-platform installation specification. The fresh installation was performed from the six requested pins, then the resolved list was recorded; reinstallation from the exported 17-package list has not been separately tested. This environment is not a complete specification for HMOG model training.

The [installation report](records/environment/install-report.json) preserves resolved versions, download-wheel hashes, package metadata, platform details, and pip version. The [installation log](records/environment/install.log) is a path-sanitized copy. No installed binaries, package caches, or font files are included in this evidence bundle.

## Preservation and privacy

All 19 selected source files from the uploaded verification archive are preserved under `records/`, together with its collection notes and original manifest. Only a private home-directory prefix in five logs was replaced with `<COMPUTE_B_HOME>`; all suffixes, dates, values, warnings, and exit-status lines were retained. Such log commands are archival context, not copy-and-paste instructions.

The original [collection manifest](records/ORIGINAL_COLLECTION_MANIFEST.sha256) records the unmodified upload identities. It is intentionally not a checksum manifest for the sanitized layout. [PACKAGING_NOTES.md](PACKAGING_NOTES.md) provides every original-to-packaged path/hash relationship; [PACKAGED_MANIFEST.sha256](PACKAGED_MANIFEST.sha256) verifies the prepared files using paths relative to the repository root.

## Scope still open

This evidence addition does not establish full retraining, dataset access or redistribution permission, clean-install reproducibility of every workflow, general test-suite success, manuscript figure reproduction, M9 user-level inference, drift-subset or Touchalytics reanalysis, full-cohort repaired-detail assembly, or final submission readiness. Preserve these limits when integrating the evidence into `docs/REPRODUCING.md`, `docs/PAPER_CODE_MAP.md`, and `PROVENANCE.md`; this supplement must not be used to mark unrelated pending tasks complete. The prior blanket descriptions of these specific three reanalyses as unexecuted should be updated in the final consolidated documentation pass, not used as a reason to repeat them.

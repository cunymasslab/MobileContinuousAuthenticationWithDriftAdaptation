# Matched-stream and M9 reference bundle: packaging notes

## Status and origin

This directory packages the maintainer-supplied `tifs_matched_stream_m9_bundle.zip`.
Original archive SHA-256: `d391058fad0f4ca1a2ea6ddcf2ee201e1f71e8c417627fa0c746c90d4885c9a7`.
The original collection notes report collection at
`2026-10-04T00:05:13.231035+00:00` on Compute B. This is the timestamp recorded
by the maintainer's collection operation, not an inferred experiment date.

The supplied archive contained 18 source files plus its collection manifest
and collection notes. All 18 source-file SHA-256 values were independently
recomputed from the uploaded archive and matched its collection manifest.
The original uploaded archive was not modified. No training, statistical
analysis, or figure-generation script was executed during this inspection.

This is an import-ready **reference and historical-source bundle**, not a
claim that a full replication workflow has been tested. Do not run a script
inside this reference directory: preserved scripts can write to fixed paths.
Keep reference inputs separate from regenerated outputs.

## What was preserved and changed

- All eight CSVs are byte-for-byte unchanged, including failure records,
  missing entries, original row ordering, and numeric serialization.
- All five Python files and `validation_users.txt` are byte-for-byte unchanged.
  They were moved under `source_snapshot/` so importing this bundle does not
  replace active repository source or the active cohort file.
- The saved `results_confound/analysis_confound.txt` is byte-for-byte unchanged.
  Its convergence warnings, diagnostic qualifications, and displayed numbers
  have not been removed or corrected.
- Both saved M9 figures are byte-for-byte unchanged, including their metadata.
- Only `results_m9_lr_sweep/sweep.log` was sanitized: 180 occurrences
  of the maintainer's absolute server-home prefix were replaced with
  `<COMPUTE_B_HOME>/`. Relative paths, source line numbers, training output,
  failures, numerical values, warnings, and log ordering were retained.
- `COLLECTION_NOTES.txt` is unchanged. The original collection manifest is
  preserved below. The current `COLLECTION_MANIFEST.sha256` validates the
  files and paths in this prepared directory, including these notes, except
  the manifest itself.

The original private prefix is deliberately not reproduced here. The mapping
must remain in the maintainer's private records. This packaging edit does not
redact earlier repository history or constitute an all-repository security audit.

## Structural checks performed on the supplied CSVs

Counts below were read directly using Python's CSV parser during packaging;
they are not results of a new model or statistical run. “Nonempty EER” is
specific to the corresponding model's overall-EER field, not a claim that
all metrics in that row are populated.

| Artifact | Data rows | Target users | Duplicate configuration rows | Nonempty model overall EER |
|---|---:|---:|---:|---:|
| `results_confound/results_confound_summary.csv` | 9396 | 88 | 0 | Not applicable |
| `results_confound/results_confound_failed.csv` | 108 | 6 | 0 | Not applicable |
| `results_m9_lr_sweep/comparison.csv` | 5 | Not applicable | Not applicable | Not applicable |
| `results_m9_lr_sweep/M2_lr_0.002_summary.csv` | 360 | 10 | 0 | 330 |
| `results_m9_lr_sweep/M9_lr_0.0001_summary.csv` | 360 | 10 | 0 | 330 |
| `results_m9_lr_sweep/M9_lr_0.0005_summary.csv` | 360 | 10 | 0 | 330 |
| `results_m9_lr_sweep/M9_lr_0.001_summary.csv` | 360 | 10 | 0 | 330 |
| `results_m9_lr_sweep/M9_lr_0.002_summary.csv` | 360 | 10 | 0 | 330 |

The configuration key used for the structural check was
`(target_user, train_condition, stream_condition, seed)`.
The matched-stream summary and failure records have disjoint key sets.
The five sweep summaries share the same 330 keys with nonempty, finite model
overall-EER values. No rows were removed, added, reordered, or filled in.
The saved `comparison.csv` was inspected as a five-row reference table; its
numerical aggregation was not regenerated during packaging.

## Source and execution boundaries

The five supplied Python files parse successfully as Python source. They have
not been imported or executed here. A source snapshot's presence does not
by itself establish the exact producing commit, original environment, or
reproduction of its saved output.

The matched-stream report contains user-weighted results for 88 users and
separately labeled user-by-category diagnostics. Preserve those distinctions.
Its mixed-effects section includes boundary and non-positive-definite Hessian
warnings; they remain in the unchanged report. Displayed p-values formatted
as zero must not be interpreted as exact zero probabilities.

`source_snapshot/analyze_confound.py` writes the report and figures under a
fixed `results_confound/` directory. Its final baseline comparison also looks
for `results_stage4_merged_summary.csv` in the working directory. That file is
not in this collection. The accepted repaired factorial summary is a separate
reference artifact already packaged elsewhere; any use as a replacement input
must be explicit and verified, not silently renamed or assumed identical.

The preserved matched-stream source has two comment/implementation distinctions
that remain unchanged here: the six-test Holm description is followed by
separate three-value Pearson and Spearman adjustments, and the mixed-effects
text describes a random intercept while the call also specifies a stream-type
random slope. These are static observations, not new statistical results or
a reason to rewrite the original saved report. A subsequent executable
reproduction must document the calculation actually used.

`source_snapshot/run_m9_lr_sweep.py` includes the `write_comparison` aggregation
routine as well as dataset-backed training. Do not execute the training driver
merely to reconstruct a summary. The supplied M9 figure renderer reads the
saved comparison table and writes to a fixed `figures/` directory.
The archive does not contain a separate final M9 user-level inference script
or its saved confidence-interval report. That specific reproduction route
must be connected to the retained earlier evidence; it is not established
by the five-row descriptive comparison alone.

## Privacy inspection scope

The text payloads were screened for server-home paths, known server labels,
IPv4-shaped strings, user-at-host patterns, common token shapes, and private-key
headers. After the log-prefix substitution, no such remaining matches were
found. PNG/PDF metadata and the rendered one-page figure were inspected; no
personal-path or credential content was observed. This is a bounded packaging
inspection, not a guarantee that all confidential information or dataset-rights
questions have been resolved. The scientific participant codes are retained;
no direct-identifier substitution or dataset anonymization is claimed.

## Original-to-packaged file identities

| Original path | Prepared relative path | Original bytes | Original SHA-256 | Prepared bytes | Prepared SHA-256 |
|---|---|---:|---|---:|---|
| `results_confound/results_confound_summary.csv` | `results_confound/results_confound_summary.csv` | 4019780 | `f1a5f9843df2d2a59fafcddc191eab41a8da12a0f551070c49422e6548047bb1` | 4019780 | `f1a5f9843df2d2a59fafcddc191eab41a8da12a0f551070c49422e6548047bb1` |
| `results_confound/results_confound_failed.csv` | `results_confound/results_confound_failed.csv` | 14566 | `de0f95a336d7ce215b632577102fc64e9e3a2da428fdfefbb3ad416c8ce1e1d1` | 14566 | `de0f95a336d7ce215b632577102fc64e9e3a2da428fdfefbb3ad416c8ce1e1d1` |
| `results_m9_lr_sweep/comparison.csv` | `results_m9_lr_sweep/comparison.csv` | 486 | `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8` | 486 | `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8` |
| `results_m9_lr_sweep/M2_lr_0.002_summary.csv` | `results_m9_lr_sweep/M2_lr_0.002_summary.csv` | 113820 | `db0472d3736f5103ae812a842d8ea1dd5874546a95a89eb555c1a84c95b69ca1` | 113820 | `db0472d3736f5103ae812a842d8ea1dd5874546a95a89eb555c1a84c95b69ca1` |
| `results_m9_lr_sweep/M9_lr_0.0001_summary.csv` | `results_m9_lr_sweep/M9_lr_0.0001_summary.csv` | 115687 | `5d63ecaf9c446ed88a213f36e8b404ec36ed84fcb69e4db3bb51d7e034a59505` | 115687 | `5d63ecaf9c446ed88a213f36e8b404ec36ed84fcb69e4db3bb51d7e034a59505` |
| `results_m9_lr_sweep/M9_lr_0.0005_summary.csv` | `results_m9_lr_sweep/M9_lr_0.0005_summary.csv` | 115202 | `1f12d7e74201fff83a92a11fc5b479bedec25f0c1dd617b589d0dee1e39bb609` | 115202 | `1f12d7e74201fff83a92a11fc5b479bedec25f0c1dd617b589d0dee1e39bb609` |
| `results_m9_lr_sweep/M9_lr_0.001_summary.csv` | `results_m9_lr_sweep/M9_lr_0.001_summary.csv` | 115257 | `561393f61b822771938d87ea35d9b8ead275183cfff61a01ef11dc15927b97b7` | 115257 | `561393f61b822771938d87ea35d9b8ead275183cfff61a01ef11dc15927b97b7` |
| `results_m9_lr_sweep/M9_lr_0.002_summary.csv` | `results_m9_lr_sweep/M9_lr_0.002_summary.csv` | 114114 | `f16fe8cdeaa1e4b2d23f122953c8eadfe6fa5ed76193a42f2c70d0c9015bdae6` | 114114 | `f16fe8cdeaa1e4b2d23f122953c8eadfe6fa5ed76193a42f2c70d0c9015bdae6` |
| `results_confound/analysis_confound.txt` | `results_confound/analysis_confound.txt` | 10971 | `90fb8051bbba1d027e2ec289553c28f8640dc5c7c2d5f0997a447bb967028295` | 10971 | `90fb8051bbba1d027e2ec289553c28f8640dc5c7c2d5f0997a447bb967028295` |
| `results_m9_lr_sweep/sweep.log` | `results_m9_lr_sweep/sweep.log` | 3169477 | `ebc7a5d8b9b0258363d6f1c97469ee14268d12e61206eca68d535a06ffe827ef` | 3170017 | `b2b6680703dbce037a5f0dfff3a2830409a2f2aaec444cac0e80f351b7306ca3` |
| `analyze_confound.py` | `source_snapshot/analyze_confound.py` | 22509 | `2d215f1813434c2784c7fb4efa36842d7c1bb6bf2dd2964d9204ef61cb0a15fe` | 22509 | `2d215f1813434c2784c7fb4efa36842d7c1bb6bf2dd2964d9204ef61cb0a15fe` |
| `generate_confound_figures.py` | `source_snapshot/generate_confound_figures.py` | 13716 | `987a52041aa549562227b677e125e73f303871bba9c5a213ddf900c0930adb06` | 13716 | `987a52041aa549562227b677e125e73f303871bba9c5a213ddf900c0930adb06` |
| `run_confound.py` | `source_snapshot/run_confound.py` | 25218 | `fc73ed46da5c9880640780621ae33a7426a2481281f1684b26311976e5c4f40d` | 25218 | `fc73ed46da5c9880640780621ae33a7426a2481281f1684b26311976e5c4f40d` |
| `run_m9_lr_sweep.py` | `source_snapshot/run_m9_lr_sweep.py` | 11608 | `b9499b0785d925440efaa60506f326776933782fc8d01ae421d3275bf4abb5f2` | 11608 | `b9499b0785d925440efaa60506f326776933782fc8d01ae421d3275bf4abb5f2` |
| `generate_m9_sweep_figure.py` | `source_snapshot/generate_m9_sweep_figure.py` | 5519 | `3ea1beb456dc7a8abb915c7b12ba7a957beb36ac094f84114e4c803151b23748` | 5519 | `3ea1beb456dc7a8abb915c7b12ba7a957beb36ac094f84114e4c803151b23748` |
| `validation_users.txt` | `source_snapshot/validation_users.txt` | 70 | `e36af2ac3535f29ba0dea48ef62b49f45a6b1300c36d357ba510a305300cdb0f` | 70 | `e36af2ac3535f29ba0dea48ef62b49f45a6b1300c36d357ba510a305300cdb0f` |
| `figures/m9_lr_sweep.png` | `figures/m9_lr_sweep.png` | 57369 | `51509aa11bf52066a7b2692594c77363bbfbb0831d1f17929cc525c3cbf74adb` | 57369 | `51509aa11bf52066a7b2692594c77363bbfbb0831d1f17929cc525c3cbf74adb` |
| `figures/m9_lr_sweep.pdf` | `figures/m9_lr_sweep.pdf` | 16077 | `308bd16a648112be7323afb363f40281e813a8c72302279066941611e1490473` | 16077 | `308bd16a648112be7323afb363f40281e813a8c72302279066941611e1490473` |

## Original collection manifest, preserved verbatim

The following block records paths in the **original uploaded archive**. Do not
run it as a validation manifest for the relocated/sanitized directory.

```text
f1a5f9843df2d2a59fafcddc191eab41a8da12a0f551070c49422e6548047bb1  results_confound/results_confound_summary.csv
de0f95a336d7ce215b632577102fc64e9e3a2da428fdfefbb3ad416c8ce1e1d1  results_confound/results_confound_failed.csv
ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8  results_m9_lr_sweep/comparison.csv
db0472d3736f5103ae812a842d8ea1dd5874546a95a89eb555c1a84c95b69ca1  results_m9_lr_sweep/M2_lr_0.002_summary.csv
5d63ecaf9c446ed88a213f36e8b404ec36ed84fcb69e4db3bb51d7e034a59505  results_m9_lr_sweep/M9_lr_0.0001_summary.csv
1f12d7e74201fff83a92a11fc5b479bedec25f0c1dd617b589d0dee1e39bb609  results_m9_lr_sweep/M9_lr_0.0005_summary.csv
561393f61b822771938d87ea35d9b8ead275183cfff61a01ef11dc15927b97b7  results_m9_lr_sweep/M9_lr_0.001_summary.csv
f16fe8cdeaa1e4b2d23f122953c8eadfe6fa5ed76193a42f2c70d0c9015bdae6  results_m9_lr_sweep/M9_lr_0.002_summary.csv
90fb8051bbba1d027e2ec289553c28f8640dc5c7c2d5f0997a447bb967028295  results_confound/analysis_confound.txt
ebc7a5d8b9b0258363d6f1c97469ee14268d12e61206eca68d535a06ffe827ef  results_m9_lr_sweep/sweep.log
2d215f1813434c2784c7fb4efa36842d7c1bb6bf2dd2964d9204ef61cb0a15fe  analyze_confound.py
987a52041aa549562227b677e125e73f303871bba9c5a213ddf900c0930adb06  generate_confound_figures.py
fc73ed46da5c9880640780621ae33a7426a2481281f1684b26311976e5c4f40d  run_confound.py
b9499b0785d925440efaa60506f326776933782fc8d01ae421d3275bf4abb5f2  run_m9_lr_sweep.py
3ea1beb456dc7a8abb915c7b12ba7a957beb36ac094f84114e4c803151b23748  generate_m9_sweep_figure.py
e36af2ac3535f29ba0dea48ef62b49f45a6b1300c36d357ba510a305300cdb0f  validation_users.txt
51509aa11bf52066a7b2692594c77363bbfbb0831d1f17929cc525c3cbf74adb  figures/m9_lr_sweep.png
308bd16a648112be7323afb363f40281e813a8c72302279066941611e1490473  figures/m9_lr_sweep.pdf
```

## Current manifest

Use the standalone `COLLECTION_MANIFEST.sha256` in this directory to check the
prepared files. It covers 20 files: the 18 collected source files at their
prepared paths, the original collection notes, and these packaging notes.
The current manifest itself is not self-hashed.

# Packaging notes: local drift and Touchalytics diagnostic references

Prepared on 2026-10-04 for a separate, additive reference-artifact import.
This is a file-identity and structural inspection, not a rerun of the experiments.

## Input and disposition

Input archive: `tifs_local_drift_touchalytics_bundle.zip`.

Input archive SHA-256:

```text
fea1035eebe1ba97e199d5dd0c08d0cd7348955a958bf7108080ad3b97f05792
```

The input contains 12 source files (four CSVs, seven Python files, and one
configuration list), plus collection notes and a collection manifest: 14 entries.
All 13 entries named by its manifest match their recorded SHA-256 values.
The manifest itself is not self-hashed.

Every original archive entry is preserved byte-for-byte under this directory,
including both collection metadata files. No source-code line, CSV cell,
filename below this directory, or collection record was edited or removed.
No privacy substitutions were necessary for this batch after the scoped scan
below. The only additions are this file and `PACKAGED_MANIFEST.sha256`.
The prepared ZIP contains 16 files, all below `artifacts/reference/drift_touchalytics_diagnostics/`.
The uploaded archive remains unchanged.

## Historical file identities reconciled

The hashes below were recomputed from the uploaded CSV bytes and compared with
the existing inventory records. All four match. This links the supplied copies
to those historical observations; it does not newly establish scientific validity
or identify the exact producing execution solely from a hash.

| Relative file | Bytes | Data rows | Recomputed SHA-256 | Retained record |
|---|---:|---:|---|---|
| `results_stage4b_exp1/results_drift_detection.csv` | 4716 | 50 | `d21cc7f8f85da60aac8c1c5725e41aed655ccb6395102f643b026a402628babe` | Audit manifest, row 141 |
| `results_stage4b_exp1/results_stage4b_cv_summary.csv` | 7560 | 50 | `5c1bf57c15965e0b0bbc7dc4d07bc2f0b7c74d2dcb07b16a71fb797bf6c23b1f` | Audit manifest, row 142 |
| `touchalytics_data/hmog_touchalytics_comparison.csv` | 234 | 3 | `fb9056fc769739184d2b61b7d68475ddf1c86010fcc8adc427dd7e5b40332852` | Retained Compute A transcript, lines 256-264 |
| `touchalytics_data/touchalytics_drift_results.csv` | 7091 | 68 | `644d00486f5f1d3cf53b2ebae3672689baaaf69434f2edfdb621879fbcaeb0c6` | Retained Compute A transcript, lines 272-278 |

Audit-manifest source: `avinashjairam/rtrl_from_scratch`,
`audit/manifest.csv` at commit
`5a288190072f2fea91c67d7b9a7dd803f4d3f26a` (August 25 audit).
Its Git blob ID is `699a37e3d0b15e0c28a84fcfeead7a47deb044a6`.
The two cited rows record transfers from Compute A and local inspection.

Compute A transcript source: `Pasted text(20260826-230140).txt`.
Line references above refer to that transcript's original file lines, not
line numbers within a retrieval response. The full transcript is not redistributed
here. It records the Touchalytics comparison file's blank HMOG fields as well
as the identities, sizes, and row counts above.

## Structural inspection of the supplied data

Both HMOG CSVs contain 50 unique `(target_user, train_condition,
stream_condition)` keys and the same 42 target users. Their key sets agree with
one another and with `source_snapshot/exp1_50_users.txt`. The CV summary has
only seed 42; the drift-detection CSV has no seed column.
Their stored `drift_gap` strings match exactly after joining by those keys.
No empty cells or duplicate keys were found in either HMOG CSV.

Applying the category definition already present in the collected figure
source to the recorded condition labels yields 10 temporal, 20 cross-activity,
and 20 cross-both combinations. There are no cross-posture combinations in
this selected subset. These are combination counts, not counts of distinct
people in disjoint categories.

The Touchalytics drift CSV contains 68 unique `(user_id, drift_type)` keys
across 41 user IDs: 13 `temporal`, 41 `cross_activity`, and 14 `cross_both`
rows. No empty cells or duplicate keys were found. Its recorded values,
including negative MMD estimates, are retained unchanged; no clipping or
recalculation was performed during packaging.

The three-row `hmog_touchalytics_comparison.csv` contains nine blank HMOG
cells (three columns across three metric rows). They remain blank.
`source_snapshot/touchalytics_hmog_comparison.py` explicitly leaves HMOG
values blank for manual entry. This file is therefore not a completed
HMOG-versus-Touchalytics numeric comparison. It supplies Touchalytics summaries
only; packaging does not fill the HMOG cells or infer missing results.

## Source snapshots and execution status

The seven Python files were parsed for syntax and inspected statically for
paths and output behavior; they were not imported or executed. Their syntax
parsed successfully. Their placement under `source_snapshot/` does not replace
active repository scripts, and it does not establish them as the exact revisions
that produced the historical experiments.

These are not self-contained executable instructions. Some snapshots import
other project modules and require raw data. The historical figure scripts write
to `figures/` relative to the working directory. The Touchalytics comparison
script writes its CSV into the supplied data directory and can overwrite a file
with that name. Do not run it in this preserved reference folder.
The included `analyze_stage4b.py` is a historical source snapshot, not a claim
that it is the final analysis route for the two included Experiment 1 CSVs.

No historical model training, source-classifier fitting, KS calculation,
statistical inference, summary-program execution, or figure rendering was rerun
while preparing this archive. Structural checks and checksum comparisons were
performed only on the supplied files.

## Scoped privacy inspection

All 14 original archive entries were read as UTF-8 text and scanned for personal
home-directory prefixes, IPv4 addresses, user-at-host strings, known private
host labels, private-key headers, common GitHub token shapes, and AWS access-key
shapes. No matches were found. This is a limited pattern-based inspection of
this bundle, not a full repository/history security review or data-license review.
Dataset user identifiers, result values, and configuration keys were preserved.
No raw sensor data, feature matrix, credentials, shell history, or environment
binaries were included by the explicit collection list.

## Remaining scope

The default Touchalytics adaptation results, raw-event adaptation results,
and sweep/run logs are not in this bundle. The two diagnostic CSVs are not
substitutes for those artifacts. Figure reproduction and execution of the
artifact-based summary routes have not been completed by this import.
Preserve prior verification records and their original scope; do not restart
completed experiments merely because remaining artifacts need integration.

## Manifests

`COLLECTION_MANIFEST.sha256` remains unchanged and covers the original
12 source files plus the original collection notes, relative to this folder.
`PACKAGED_MANIFEST.sha256` covers all other 15 prepared files, including the
unchanged collection manifest and these packaging notes, relative to this folder.
Run a manifest check from this folder only; neither manifest includes itself.

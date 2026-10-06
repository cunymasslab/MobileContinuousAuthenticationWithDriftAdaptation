# Packaging notes: Touchalytics adaptation reference files

Prepared 2026-10-04 for an additive reference-artifact import.
This records file identity, structural inspection, and a limited privacy substitution.
It is not a new execution of the original experiments or statistical analyses.

## Input archive and contents

Input: `tifs_touchalytics_adaptation_bundle.zip`.
Input SHA-256: `18ec8b23ace01ec1f8be7f8fecb71624566671a0ecedcb84466aca3553426581`.
Collection timestamp recorded in the input notes: `2026-10-04T20:15:55.952133+00:00`.
Collection location: Compute A.

The original ZIP has 13 entries: two adaptation-result CSVs, two logs, six Python
source snapshots, the historical dataset readme, collection notes, and a collection
manifest. All 12 files named in `COLLECTION_MANIFEST.sha256` matched their recorded
SHA-256 values before packaging. The manifest does not include itself.

The prepared ZIP has 15 files under `artifacts/reference/touchalytics_adaptation/`.
All original relative filenames are retained. The only edited original file is
`touchalytics_run.log`, as detailed below. The two additions are these notes and
`PACKAGED_MANIFEST.sha256`. The uploaded original ZIP has not been edited.

## Historical artifact identities

The four original artifact hashes below were recomputed from the uploaded bytes
and match the retained Compute A inventory in `Pasted text(20260826-230140).txt`.
Source-line references in this table are original transcript lines, not retrieval
response line numbers. The full private transcript is not redistributed here.

| Original relative file | Bytes | Original SHA-256 | Historical source lines |
|---|---:|---|---|
| `results_touchalytics/touchalytics_adaptation_results.csv` | 27591 | `d80f8c2aac5c5cc874a9e91b55c8cf3d8cc90c075fa8ad820aa30d0680bb3636` | 155-162 |
| `results_touchalytics_raw/touchalytics_raw_adaptation_results.csv` | 29557 | `7f3da80dd06c59795fbd4155f0fe642111142f7d1898d65ecb775991ab3ae212` | 179-185 |
| `touchalytics_sweep.log` | 368919 | `d40753430b2f576a7da51847ed7cb53b675775a1de581f67122c514b98bba3c5` | 324-327 |
| `touchalytics_run.log` | 345522 | `b1d62090c0ff2067d3d5f7997795b295b28e573d3090ba3eb987bc4c272a2ae4` | 319-322 |

These matches connect the supplied bytes to the prior recorded inventories.
They do not by themselves establish scientific correctness, a producing source
revision, or full experimental reproducibility.

## CSV structural checks

Both CSVs have unique `(user_id, drift_type, seed)` keys, 40 distinct user IDs,
and all three seeds (`42`, `123`, `456`) for every included user/drift pair.
Neither has blank cells, duplicate keys, or malformed row widths. The recorded
M1-M8 EER cells parse as finite values in [0, 1]. No value was changed.

| File family | Rows | Columns | Temporal rows/users | Cross-activity rows/users | Cross-both rows/users |
|---|---:|---:|---:|---:|---:|
| Default historical result | 180 | 14 | 18 / 6 | 120 / 40 | 42 / 14 |
| Separate raw-event result | 204 | 13 | 42 / 14 | 120 / 40 | 42 / 14 |

The category names stored in the CSVs are `temporal`, `cross_activity`, and
`cross_both`. Category-specific user counts overlap; do not sum them as distinct
participants. The default file additionally contains `adaptation_benefit`.
The 180-row and 204-row files are separate experiments, not alternate filenames
for the same output. No inference, EER recomputation, or model training was run
as part of these structural checks.

## Logs and selection evidence preserved

`touchalytics_run.log` has 4,822 lines. It records 1,980 loaded windows of shape
(10, 30), six temporal eligible users, 40 cross-activity eligible users, and 14
cross-both eligible users. Its completion line reports 180 jobs, zero skipped,
and zero failed. Those are statements in the saved log, not a new execution.
The overflow and invalid-value RuntimeWarnings at lines 527 and 529 remain
present. The recorded EER cells being finite does not resolve those warnings.

The historical log also uses the heading "RTRL ≡ BPTT equivalence" in its final
summary. That historical wording is retained in the log (including its original
symbol); it is not a new formal equivalence-test conclusion from this package.

`touchalytics_sweep.log` has 5,316 lines and is unchanged. It preserves all four
sequential rounds (window size, online learning rate, feature set, and hidden
size), not just the selected result. The final configuration printed at lines
5311-5316 is W=50, online learning rate 0.002, feature set `raw_dyn6`, H=32.
The selected row at lines 5303-5305 prints n=40, M1 EER 0.3677, M4 EER 0.3479,
benefit +0.0198, and Wilcoxon p=0.0223. These are copied log observations at the
precision printed there; no statistical calculation was rerun.

The collected sweep source's `select_winner` function prioritizes the largest
significant positive benefit, otherwise the lowest M4 EER. Its use of the
observed results in selection must remain explicit: the selected result is
exploratory and post-selection, not an independent confirmatory evaluation.
The W=100 row reports n=37 while other window-size candidates report n=40;
that coverage variation remains visible in the full log.

The default and raw-event CSVs are not the selected-sweep CSV. No per-user
selected-sweep output is included in this collection. Do not manufacture such a
file or replace either default result with the selected configuration's means.
This packaging step does not newly establish the reported nonsignificance of
any default or raw-event comparison; their source data are preserved for the
separate analysis/evidence integration route.

## Source snapshots and filename/version limitation

All six Python snapshots remain byte-for-byte unchanged. They were parsed for
syntax and inspected as text, not imported or executed. They remain isolated
under `source_snapshot/` and do not replace the repository's active source files.

The collected `run_touchalytics_adaptation.py` cannot be assumed to be the exact
producing version of the historical default result. It imports the raw-event
loader, sets `INPUT_DIM=5` and `WINDOW_SIZE=50`, and names its output
`results_touchalytics/touchalytics_adaptation_raw_results.csv`. In contrast,
the saved default run log describes windows of shape (10, 30), and the retained
result is `results_touchalytics/touchalytics_adaptation_results.csv`.
Some comments and input-file checks in that source still refer to stroke
features. This evidence is retained as found, not silently reconciled.

The alternate-name file was optional and was not found during collection.
Its absence is recorded in `COLLECTION_NOTES.txt`. No alternate file was created,
no result was renamed to fit the snapshot, and no producing revision is asserted.

The raw-event driver writes under `results_touchalytics_raw/` and uses the
corresponding raw-event result basename. The analysis snapshot accepts `--csv`
and `--no-plots`; its default plotting route can write output. Both training
drivers can append results and use existing rows for resume decisions. Do not
execute the snapshots from this preserved reference folder. A separately
reviewed, isolated command is needed for any later reproduction.

The historical `touchalytics_data/readme_data.txt` is retained unchanged with its
original attribution and dataset-field descriptions. It is not a software license
or a completed data-redistribution review. Raw touch events and feature matrices
are not included in this collection.

## Privacy substitution and manifest semantics

Only three occurrences of the private project-directory prefix in
`touchalytics_run.log` were replaced with `<compute-a-project>`, on original
lines 1, 527, and 529. Filename suffixes, warnings, values, line order, and all
other log content remain unchanged. The full private prefix is deliberately
not restated here. No private-path substitution was needed in the sweep log.

| Edited file | Original SHA-256 | Packaged SHA-256 |
|---|---|---|
| `touchalytics_run.log` | `b1d62090c0ff2067d3d5f7997795b295b28e573d3090ba3eb987bc4c272a2ae4` | `b714739f6a1ea6bbb6d342425dc836e2b9a8b17fad4c840dd4a93e7d2fb45f6f` |

The original `COLLECTION_MANIFEST.sha256` remains byte-for-byte unchanged.
It records the original collection, so its `touchalytics_run.log` entry is
expected not to match the sanitized copy. Do not change that original entry
or describe a check of the original manifest as passing for all packaged files.
All its other 11 entries still match the corresponding packaged files.

Use `PACKAGED_MANIFEST.sha256` for the prepared directory: it covers all other
14 files, including the original collection manifest and these packaging notes,
relative to this directory. Neither manifest hashes itself.

A scoped scan of every original and prepared entry checked personal home paths,
IPv4 addresses, user-at-host strings, known private host labels, private-key
headers, common GitHub-token shapes, AWS access-key shapes, and simple quoted
credential assignments. The only observed private-path matches in the original
files were the three run-log prefixes described above. The prepared entries have
no matches for those patterns. This is not a complete repository/Git-history
security review, nor a licensing or participant-privacy clearance.

## Scope of this import

This recovers the two outstanding adaptation-result CSVs and the two saved logs,
and preserves available supporting snapshots. No additional server retrieval is
needed for those four artifacts. It does not authorize deleting the server or
its data, replace a broader backup check, or complete every replication task.

No project module, summary function, statistical test, experiment, or figure
renderer was executed during preparation. No p-values, model scores, or published
results were adjusted. The input ZIP and all original server files remain outside
this prepared copy and were not modified by packaging.

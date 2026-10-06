# M7-M8 analysis-source recovery

## Status and limits

Candidate 1 contains a complete recovered statistical-analysis implementation whose
input identity, output names, report structure, and overall output-field order
correspond to the supplied saved M7-M8 reference bundle. This is a source-recovery
and static-packaging result. Neither candidate nor the packaged runner has been
executed in this step. Reproduction of the stored numerical outputs is pending.
A shell-history entry is evidence of recorded command text, not by itself proof
of successful execution or an independently verified execution date.

Candidate 2 is the recent reference-ZIP creation command, not the statistical
analysis. It remains preserved in the original private uploaded ZIP and is not
included in this scientific-source package. No full shell history is included.

## Recovery sources and identities

Input archive: `tifs_m7_history_candidates.zip`

| Item | SHA-256 |
|---|---|
| Input ZIP | `6190cdbe55087ee9ab648136813b0d90e102bfdbc269cb21fe55d319d39407a6` |
| Candidate 1, complete excerpt | `49f85f8069a59e7d432bd49e6c667ef097307c406b34d03de4dd2d5d90555b8e` |
| Candidate 2, complete excerpt | `55e6edce3130e2a6e84b1e087a317205b2aecdd5ca6ccd95fea96f92f11e69d6` |
| Candidate 1 original Python body, wrapper excluded | `35e5f9bf2991e94d189cc20da48699f90597725919541d80f03af93f1a845674` |
| Sanitized recovered body | `2f80efd250bdf7468d6895cd9aa16d6f010a954e98742fe5eb1d9684cc3ba17b` |
| Packaged runner | `c9d6f4b1364748a2fc80d9c11894a6e3f654cbf0d40bad1cf5f2d0ebddb11fc7` |

Both complete-excerpt hashes match the maintainer's previously supplied recovery
output and the archive's recovery notes. They were recomputed on the uploaded
excerpts, not on a live server history file.

Candidate 1 excerpt: `candidate_1_history_lines_865-1920.txt`.
Its first and last lines are the shell wrapper; its Python body is recorded
history lines 866-1919 (1,054 lines). History line numbers refer to the original
recovered snapshot and should not be used against a subsequently changed history.

## Files in this package

- `analysis/reproduce_m7_m8_statistics.py`: a standalone packaging draft with
  explicit `--input` and `--output-dir` arguments and a main-entry guard.
- `docs/evidence/m7_analysis_recovery/recovered_analysis.py.txt`: archival text
  of the recovered body, with one private home-path prefix replaced by
  `<PRIVATE_HOME>`. Do not execute this text file.
- This file: source relationships, static findings, and packaging-change record.

The uploaded history ZIP was not modified. The archival text is NOT byte-identical
to the original history excerpt: its shell wrapper was removed and one private
path prefix was sanitized. The distinct hashes above preserve that distinction.

## Identified input and outputs

The recovered source checks the summary against this exact SHA-256 before analysis:

`7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098`

Summary: `results_stage4_merged_summary_m7_repaired_20260901.csv`.
This source constant matches the SHA-256 recomputed from that CSV in the previously
uploaded original reference bundle. No statistics were recomputed in making this
identity comparison.

The output filenames are:

```text
m7_m8_statistical_closure.txt
m7_m8_overall_statistics.csv
m7_m8_by_category.csv
m7_m8_user_level_values.csv
m7_m8_user_category_values.csv
MANIFEST.sha256
```

The 29 returned fields in `summarize_user_view` match the field order in the saved
`m7_m8_overall_statistics.csv` header. Filename/header agreement is not a numerical
reproduction result.

## Recovered analysis definitions

The references below are line numbers in the saved history snapshot, not line
numbers in the packaged runner.

| Component | Recovered behavior | History lines |
|---|---|---|
| Input identity | SHA-256 equality check against the accepted summary | 883-891, 1265-1275 |
| Output protection | Refuses an existing output directory | 1277-1283 |
| Metrics | M6, M7, and M8 overall EER columns | 911-915 |
| Numeric conversion | `pd.to_numeric(errors="coerce")` for those three columns | 1314-1321 |
| M7-M8 analysis row selection | Rejects partially populated M6/M7/M8 rows; retains rows where all three converted overall-EER fields are nonmissing | 1344-1357 |
| Coverage assertions | Checks 9,558 candidate rows, no duplicate configuration keys, 9,216 retained rows, and 89 target users | 1334-1341, 1359-1369 |
| Configuration-weighted view | Overall configuration mean and user-cluster bootstrap using sums/counts | 1399-1446 |
| Primary view | Mean across retained configurations within each user, then equal weighting of the user values | 1449-1507, 1587-1591 |
| Sensitivity view | Mean within user/category, then equal mean across four categories per user | 1510-1597 |
| Bootstrap counts | 100,000 bootstrap samples; batch size 5,000 | 925-928 |
| Sign-flip test | 200,000 samples; add-one correction | 926-928, 1026-1069 |
| Category analyses | Four user-level category comparisons, with separate Holm adjustments for each of the three test families | 1667-1709 |
| Output writing | Four statistical/user-value CSVs, one text report, and an output manifest | 1728-1750, 1785-1897 |

The recovered row-selection rule is specifically the M6/M7/M8 comparison rule.
It must not be generalized without evidence into a requirement that every M1-M8
metric, every recovery metric, or a full-detail artifact be present.
The count assertions validate the selected result; they do not trim rows to force
that result.

Recovered random seeds:
- Configuration-weighted cluster bootstrap: 42.
- Primary user-view difference bootstrap: 100.
- Equal-category sensitivity difference bootstrap: 200.
- Category-specific difference bootstraps: 1000, 1001, 1002, 1003, in the source's
  order temporal, cross-posture, cross-activity, cross-both.
- Within each view: sign-flip seed = view seed + 1000; RTRL-benefit bootstrap seed =
  view seed + 2000; BPTT-benefit bootstrap seed = view seed + 3000.

The source uses `ttest_1samp` on the per-user M7-minus-M8 differences and labels
that result as the paired test. It also calls the two-sided Wilcoxon test and the
sign-flip function. Its conclusion text is selected from computed interval bounds;
it does not insert the previously reported numerical answers as constants.

## Packaging changes and static validation

The runnable copy differs from the original in these explicitly identified ways:

1. The shell `python3` here-document wrapper is omitted.
2. A module description and `argparse` import are added.
3. The private fixed input directory is replaced by a required `--input` argument.
4. The fixed output-directory assignment is replaced by a required `--output-dir`
   argument. It must be a new directory, separate from the reference artifacts.
5. Output-path assignments and the original operational body are moved into
   `main()`, with an `if __name__ == "__main__"` entry point.

Static checks performed while preparing this package:
- The recovered body parses as Python; the packaged runner parses and compiles
  to a code object without execution.
- All seven recovered function bodies are structurally identical in the runner
  (AST comparison excluding source-location metadata).
- The nine analysis constants/lists and six output-filename assignments are
  structurally identical to their recovered definitions.
- Every original operational statement from history line 1265 onward remains
  structurally identical inside `main()` and in the same order.
- The archival text differs from the original body only in its one recorded
  private-path substitution.
- A limited scan of the three package files found no private home prefixes,
  IP-address patterns, user-at-host strings, private-key headers, or common token
  shapes. This is not a repository/history security clearance.

No code was imported from the recovered candidates. No recovered statement,
statistical calculation, shell-history command, or experiment was executed.
No file in the accepted reference bundle or the original source ZIP was edited.

## Execution boundary

Do not run the original history excerpt or change the saved reference outputs.
For a later authorized reanalysis, the draft runner requires the accepted summary
and a NEW output directory. It retains the original expected-input hash and the
original refusal to use an existing output directory. A failed run can leave a
new incomplete output directory; use a different new directory for a subsequent
attempt instead of overwriting the reference outputs.

Direct scientific imports are NumPy, Pandas, and SciPy. Their original execution
versions have not been established by this recovered excerpt. A tested reproduction
environment and comparison with the saved numerical outputs are still required.
New report paths and manifests will differ from the historical report paths/hashes;
those differences must not be hidden or treated as changed scientific results.

This package is an unexecuted source-recovery draft, not a completed replication
or an assertion that every reported number has now been reproduced.

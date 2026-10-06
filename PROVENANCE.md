# PROVENANCE — Window-Based Incremental Adaptation for Behavioral Drift in Continuous Mobile Authentication

This document records the reported relationships among source versions, experiment runs, scientific artifacts, analysis outputs, and manuscript claims.

A checksum verifies file identity; it does not by itself verify scientific correctness, completeness, or the truth of a provenance statement.

## Status vocabulary

- **Present and inspected:** available in the current checkout and examined as data.
- **Present but not validated:** available, but its scientific content or producing run has not been independently checked.
- **Manuscript-reported:** stated in the designated manuscript source supplied for this documentation task; supporting artifacts are unavailable or unverified here.
- **Historical-ledger reference:** recorded by the pre-existing, last-verified 2026-06-01 ledger; not newly verified during packaging.
- **Prior-recorded verification:** a check documented in a retained record. An integrated record establishes its stated scope, not an execution repeated during documentation work.
- **Reanalysis verified:** a retained execution and comparison establish the named artifact-based result, with any comparison exception stated. This is not full retraining.
- **Referenced but unavailable:** named by the ledger, manuscript, or source code but absent from this checkout.
- **Superseded:** retained for provenance but not used for the designated manuscript, with its reported replacement noted where possible.

The designated manuscript source was supplied as conversation text from `AVINASH___HUI___RTRL_TIFS_PAPER.zip`, original file `main.tex`, source lines 1–938. The supplied complete-file SHA-256 is:

`9bf753636ae7674f99b705c0b7699fbf79aae14cb3502b990f6e0398e850e6ba`

The archive and matching PDF are not present in this checkout. The hash and source-to-PDF match have not been independently verified here.

## Execution locations

Historical artifacts were produced or retained across multiple locations. Neutral identifiers are used in this reviewer-facing ledger:

- **Compute A:** drift cross-validation, drift detection, and Touchalytics pipeline.
- **Compute B:** confounding-variable experiment, M9 sweep, and a backup of the merged factorial summary.
- **Workstation C:** factorial summary, large per-window detail output, and figures.

Private connection details and personal filesystem paths are omitted. Maintainers should preserve any necessary private location mapping outside the public repository. This redaction does not remove prior values from Git history; public release still requires a separate history and privacy review.

## Evidence-integration policy

Prior provenance work was performed on Compute A and Compute B, and retained project records are reported to include actual terminal-output inventories, artifact hashes, schemas, counts, repair evidence, and later review or acceptance checkpoints. Their absence from this cloud checkout is an availability limitation, not evidence that the checks were never performed.

The earlier metadata excerpts remain in [`docs/evidence/PRIOR_ARTIFACT_INVENTORY.md`](docs/evidence/PRIOR_ARTIFACT_INVENTORY.md). Their availability statements describe that earlier integration, not the current checkout. The actual reference files are now indexed below; [retained reanalysis records](docs/evidence/reproduction_20261004/README.md) document subsequent executions. Do not edit historical inventories or logs to make their original dates/statuses appear current.

Before proposing a new scientific audit or rerun, reconcile those retained records with this ledger and the release package. Preserve the scope of each record: an inventory establishes what was observed at a location and time; a checksum establishes file identity; a schema or count check establishes only the fields or coverage it examined; and an acceptance checkpoint establishes only its recorded review scope. Request only a specific item that cannot be recovered from the existing records. Do not mark pending items complete merely because related evidence exists.

## Artifact references and current status

Documentation baseline: `638bb760ef572fd910cd1a9d35a1961b174c2f27` (after PR #8). This section reconciles the imported reference packages and existing execution records. No model, analysis, bootstrap, or figure-generation program was run for this documentation update. All paths below are repository-relative; each package's notes distinguish original records from sanitized copies.

| Artifact or record | Repository location | Availability and established scope | Still separate |
|---|---|---|---|
| Accepted repaired factorial summary | [`results_stage4_merged_summary_m7_repaired_20260901.csv`](artifacts/reference/m7_repaired_20260901/results_stage4_merged_summary_m7_repaired_20260901.csv) | Present; accepted identity below; 9,558 rows. Used successfully by the recovered M7–M8 reanalysis. | Remaining full-factorial table/figure routes. |
| M7 merge and acceptance records | [Reference directory](artifacts/reference/m7_repaired_20260901/) | Saved merge provenance plus summary and two-chunk detail acceptance reports are present. | Original repair input bytes, all historical launch/audit records, and full repaired-detail assembly are not all packaged. |
| Final M7–M8 outputs | [Statistical closure](artifacts/reference/m7_repaired_20260901/m7_m8_statistical_closure/) | Report, four CSVs, and derivative manifest present; retained run records exact CSV matches and a path-only report difference. | Not full model retraining or a fresh-install M7 check. |
| Matched-stream summary and failures | [Matched-stream results](artifacts/reference/matched_stream_m9/results_confound/) | 9,396 summary rows across 88 users; 108 failure rows; disjoint configuration keys documented by packaging. | Original experiment regeneration and final manuscript table/figure rendering. |
| Matched-stream saved/reproduced reports | [Execution/comparison evidence](docs/evidence/reproduction_20261004/README.md) | Execution returned 0. Only the four initial coverage-line positions differ; all other report bytes match. | Printed-precision agreement, not full-precision numeric export. Mixed-effects warnings remain. |
| Drift-verification and detection outputs | [HMOG diagnostic CSVs](artifacts/reference/drift_touchalytics_diagnostics/results_stage4b_exp1/) | Two 50-key CSVs with the same 42 users; identities match the retained audit. | Source-data regeneration, new statistical/figure reproduction, and final manuscript alignment. |
| M9 summaries and comparison | [M9 sweep results](artifacts/reference/matched_stream_m9/results_m9_lr_sweep/) | Five summaries: 360 rows each, 330 finite/nonempty model EERs each. Reporting-only comparison reproduced byte-for-byte. | Separate user-level M9 inference, training, and figure comparison. Detail/curve outputs are not included. |
| Touchalytics diagnostics | [Diagnostic results](artifacts/reference/drift_touchalytics_diagnostics/touchalytics_data/) | Drift CSV: 68 rows across 41 users. Comparison CSV: three rows with nine blank HMOG cells. Both match historical identities. | This is not a completed numerical HMOG-versus-Touchalytics table. |
| Default Touchalytics adaptation | [Historical default CSV](artifacts/reference/touchalytics_adaptation/results_touchalytics/touchalytics_adaptation_results.csv) | 180 rows across 40 users, seeds 42/123/456; original identity matched. | Collected driver is not established as the producing version. |
| Separate raw-event Touchalytics adaptation | [Raw-event CSV](artifacts/reference/touchalytics_adaptation/results_touchalytics_raw/touchalytics_raw_adaptation_results.csv) | 204 rows across 40 users, seeds 42/123/456; original identity matched. | Statistical reanalysis and source-data reproduction. |
| Touchalytics selection/execution logs | [Packaging and log record](artifacts/reference/touchalytics_adaptation/PACKAGING_NOTES.md) | Full sweep log unchanged; run log path-sanitized with original/packaged hashes documented. | Selected result remains exploratory/post-selection; no selected-sweep per-user CSV is included. |
| Full-cohort factorial detail | Historical name `results_stage4_merged_detail_with_m7.csv` | Underlying full-cohort detail is not in these reference packages. The saved acceptance report is limited to two repaired detail chunks. | Any required final repaired-detail identity/assembly must be linked to retained evidence, not inferred from the filename. |
| Historical Level-2 placeholder | `results_bptt_h32_level2.csv` | Earlier inspected header-only placeholder, not a manuscript result. | Disposition review; do not substitute it for missing outputs. |

The accepted summary SHA-256 is:

```text
7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098
```

The older `results_stage4_merged_summary_with_m7.csv` name is a historical input name, not an alias for the accepted repaired file. Its original identity and the repair replacement are retained in [MERGE_PROVENANCE.txt](artifacts/reference/m7_repaired_20260901/MERGE_PROVENANCE.txt). The main summary is not evidence of a single M1–M9 factorial: M9 is a separate exploratory sweep. Do not overwrite or relabel historical artifacts to eliminate that distinction.

### Package manifests and sanitization

Use [the reproduction guide](docs/REPRODUCING.md#reference-integrity) for the applicable manifest and working directory. M7 has regenerated derivative manifests, with originals quoted in its notes. The matched-stream package's current collection manifest describes its prepared layout. The diagnostic package has an unchanged collection manifest plus a packaged manifest. The Touchalytics original collection manifest intentionally differs for the sanitized run log; its packaged manifest validates the distributed copy. The execution-evidence manifest uses repository-relative paths. These are distinct conventions, not grounds for silently regenerating historical manifests.

## Main-factorial configuration accounting

The designated manuscript reports:

| Stage | Count | Reported meaning |
|---|---:|---|
| Specified | 9,612 | 89 target users × 36 ordered condition pairs × 3 seeds |
| Skipped before attempt | 54 | Required training or streaming data unavailable |
| Attempted | 9,558 | Configurations passed to the experiment pipeline |
| Failed: incompatible feature dimensions | 270 | Data-loading termination |
| Failed: malformed sensor CSV | 72 | Parse termination |
| Complete outputs used in performance analyses | 9,216 | Main-factorial analysis population |

**Status by scope:** the accepted summary and retained execution now establish 9,558 input rows and 9,216 retained M6/M7/M8 configurations across 89 users. The recovered rule in [`analysis/reproduce_m7_m8_statistics.py`](analysis/reproduce_m7_m8_statistics.py) converts the three overall-EER columns to numeric, rejects partially populated M6/M7/M8 rows, and retains rows where all three values are present. It checks the accepted input hash, unique configuration keys, and expected coverage. This is the actual recovered comparison rule, not a filter invented to obtain the count. It is not automatically the completeness rule for every other model, metric, or figure.

The 9,612 specified / 54 skipped accounting and 270 / 72 failure-reason breakdown above remain manuscript-reported unless separately supported by an integrated record. Do not present the M7 comparison's successful count as a fresh validation of all those causes or of full-detail coverage. The recorded reanalysis and its source are linked in [the execution evidence](docs/evidence/reproduction_20261004/README.md).

## M7 initialization audit and repaired-result provenance

The manuscript reports that the initial separate M7 run did not exactly reproduce the original frozen-GRU control for 16 target users. The package now contains saved records of the subsequent repair and its acceptance; these are no longer wholly unavailable manuscript-only claims. The original affected-user diagnosis, complete source/launch chain, and repair inputs must still be distinguished from the acceptance records actually included.

[MERGE_PROVENANCE.txt](artifacts/reference/m7_repaired_20260901/MERGE_PROVENANCE.txt) records replacement of only ten M7 metric columns for 1,728 keys from repair chunks 1 and 2: 1,656 successful configurations plus 72 historical failure stubs, covering 16 target users. It records the historical input, the two repair-summary hashes, and the accepted repaired-summary identity. It states that non-M7 values and M7 values outside the repaired set remained unchanged.

The [summary acceptance report](artifacts/reference/m7_repaired_20260901/summary_acceptance_audit.txt) records exact metadata, M6/M8 metric, and failure-key agreement. The [detail acceptance report](artifacts/reference/m7_repaired_20260901/detail_acceptance_audit.txt) records the two repaired chunks' coverage, row identities, M6/M8 score agreement, and valid repaired M7 scores. These are retained prior checks, not checks rerun for this documentation update, and not proof that a full-cohort repaired-detail file is included.

The [recovered statistical source](analysis/reproduce_m7_m8_statistics.py), [recovery notes](docs/evidence/m7_analysis_recovery/RECOVERY_NOTES.md), accepted statistical outputs, and [October execution/comparison records](docs/evidence/reproduction_20261004/README.md) are integrated. The retained comparison records byte equality for all four statistical CSVs and report equality after normalizing only the candidate-input path. The source file's old "not executed" draft header is historical text superseded by these execution records; the code was not edited to rewrite that history.

The accepted user-level report gives a mean M7-minus-M8 difference of 0.000514093992147 EER (0.051409399 percentage points). This is a small absolute, statistically detectable difference, not a formal equivalence result. Its exact numerical record is [the saved report](artifacts/reference/m7_repaired_20260901/m7_m8_statistical_closure/m7_m8_statistical_closure.txt).

Reconcile remaining source/launch or detail-assembly links only where a claimed reproduction level needs them. Do not repeat accepted scientific checks because their historical inputs are not all in this checkout. The remaining named routes and release gates are maintained in [the reproduction guide](docs/REPRODUCING.md#remaining-work-and-stop-rules).

## October 5 source-snapshot integration

A fresh source archive matching commit `a017f58ef14a31d50b30c5b9e37fd8c8a1281548` was exercised in the separately described existing environment in [the central guide](docs/REPRODUCING.md). This is package integration testing, not a new historical training/audit claim or a fresh dependency installation.

The [two-table script](analysis/reproduce_summary_tables.py), [M9 user-level script](analysis/reproduce_m9_user_statistics.py) and [Touchalytics script](analysis/reproduce_touchalytics_statistics.py) are newly reconstructed reproduction code. They read fixed-hash references; no manuscript is needed. They respectively preserve two weighting rules with subtraction before rounding; an explicitly t-based, post-selection paired-user analysis; and separate default/raw-event analyses plus sweep-log reporting. The selected-sweep p-value is not newly reproduced from absent per-user data.

The [publication variants](analysis/publication_figures/README.md) are likewise reconstructed presentation code. Relocated direct invocations completed; prior numerical-payload matches and non-identical manuscript pixels remain the scope of the retained review.

All seven distributed-layout manifests passed (163 entries). All four numerical-test programs and eleven analysis/rendering commands completed. In the newer environment, two M7 user-value CSVs remained byte-identical, but its two statistical CSVs differed in 12 last-digit cells (maximum absolute difference `3.47e-18`). Its report matched after input-path normalization. The M9 sweep CSV remained exact and the matched-stream report retained only the prior coverage-order exception. The earlier environment's four-CSV exact result is not rewritten or invalidated by that distinct compatibility test.

The integration changes affect test failure signalling and scalar extraction, plus current instructions; core model/experiment programs and historical evidence are unchanged. The original three modified test programs' successful stdout is preserved by the fixes, with numerical tolerances unchanged. Inspection and execution logs from this step remain outside the code update. No paper, manuscript source or PDF/source-check report is added.

## Historical claim ledger — last recorded verification 2026-06-01

The numerical tables below are preserved verbatim from the pre-existing historical ledger beginning with its claim tables. Their `verified` entries record that ledger's historical status; they were not newly rerun or revalidated during documentation preparation. Current manuscript alignment and current artifact availability are described above and in `docs/PAPER_CODE_MAP.md`. The historical labels, counts, source descriptions, and comparisons below are not a current release certification and do not override the scoped status above.

## Abstract / Introduction
| Claim | Value | Source | Computation | Status |
|---|---|---|---|---|
| Participants (HMOG) | 100 | Sitova et al. (ref13) | dataset fact | verified |
| Test users | 89 | summary file | 99 - 10 validation | verified |
| Experiments (main) | >9,500 | results_stage4_merged_summary_with_m7.csv | 9,558 rows | verified |
| Median drift gap | 0.32 | results_stage4b_cv_summary.csv | median(drift_gap), 50 rows | verified |
| GRU frozen->adapt EER | 0.295 -> 0.091 | summary file | mean M6 vs M7 overall_eer | verified |
| LSTM param ratio | ~4x (3.93) | architecture calc | param count M5 vs M2 at H=32 | verified |

## Related Work (verified against source PDFs)
| Claim | Value | Source | Status |
|---|---|---|---|
| Frank/Touchalytics EER | 2-3% | Frank et al. (ref20) | verified |
| TypeNet mobile EER / subjects | 9.2% / 60,000 | Acien et al. (ref25), Table 4 + abstract | verified (60k = mobile; 168k = desktop) |
| TypeFormer EER | 3.25% | Stragapede et al. (ref26) | verified |
| DAKOTA EER | 3.5% | Incel et al. (ref28) | verified |
| KedyAuth EER / FRR | 4.91% (SVM ens.) / ~2.5% FRR | Huh et al. (ref29), Table 8 | verified |
| IncreAuth long-term EER | 8.77% | Shen et al. (ref30) | verified |
| AUToSen | observes drift, recommends retrain, no adaptation impl. | Abuhamad et al. (ref32), Sec IV-E | verified |
| Elelimy/RTU convergence | T-BPTT -> RTRL as T grows | Elelimy et al. (ref31), Fig 12 / App F.1 | verified |

## Results - Drift Verification (V-A)
| Claim | Value | Source | Computation | Status |
|---|---|---|---|---|
| Median within-dist EER | 0.07 | cv_summary.csv | median(within_eer) | verified |
| Median session-4 EER | 0.42 | cv_summary.csv | median(session4_eer) | verified |
| Median drift gap | 0.32 (0.319) | cv_summary.csv | median(drift_gap) | verified |
| Temporal drift gap | 0.13 | cv_summary.csv | median, temporal subset | verified |
| Cross-condition gaps | 0.35, 0.43 | cv_summary.csv | per-category means | verified |
| % gaps > 0.50 | ~38% | cv_summary.csv | share | verified |
| % gaps < 0.10 | ~28% | cv_summary.csv | share | verified |

## Results - Drift Detection (V-B)
| Claim | Value | Source | Computation | Status |
|---|---|---|---|---|
| Mean KS range | 0.21-0.77 | results_drift_detection.csv | min/max mean_ks_stat | verified |
| Classifier acc by category | 0.91 / 0.99 / 1.00 | results_drift_detection.csv | median by category | verified |
| % acc > 0.95 | 74% | results_drift_detection.csv | share | verified |
| Corr drift_gap vs clf_acc | r=0.519, p<0.001 | cv + detection | Pearson | verified |
| Corr drift_gap vs max_ks | r=0.515, p<0.001 | cv + detection | Pearson | verified |

## Results - Adaptation Performance (V-C)
| Claim | Value | Source | Computation | Status |
|---|---|---|---|---|
| M1 overall EER | 0.314 | summary file | mean M1_overall_eer | verified |
| M2 overall EER | 0.145 | summary file | mean M2_overall_eer | verified |
| M2 beats frozen | 89.4% | summary file | share M2<M1 | verified |
| Cross-both M1 / M2 | 0.368 / 0.172 | summary file | mean, cross-both subset | verified |
| M5 (LSTM frozen) | 0.295 | summary file | mean M5_overall_eer | verified |
| M2=M4, M7~M8 | 0.145/0.145, 0.091/0.090 | summary file | means | verified |
| GRU vs Vanilla (adapt) | ~37% | summary file | (0.145-0.091)/0.145 | verified |
| Per-category M1/M2 | temporal 0.221/0.103; x-post 0.323/0.155; x-act 0.303/0.134; x-both 0.368/0.172 | summary file | per-category means | verified |
| M2 recovery | -0.217 | summary file (with_m7) | mean(M2_end - M2_initial) | verified |
| M7 recovery | -0.234 | summary file (with_m7) | mean(M7_end - M7_initial) | verified |
| M1 recovery | -0.003 | summary file (with_m7) | mean(M1_end - M1_initial) | verified |
| Recovery-curve experiments | 9,216 | detail file | experiments w/ valid rolling_eer | verified |
| Forgetting rate | 1.1% | summary file | share M2_end - M1_end > 0.05 | verified |
| Adaptation helped | 79.1% | summary file | share M2_end < M1_end | verified |
| Median forgetting delta | -0.146 | summary file | median(M2_end - M1_end) | verified |
| M3 (legit-only) EER | 0.223 | summary file | mean M3_overall_eer | verified |

## Results - Confounding Variable (V-D)
| Claim | Value | Source | Computation | Status |
|---|---|---|---|---|
| Experiments | 9,396 | results_confound_summary.csv | row count | verified |
| No-drift benefit | 0.007 | confound summary | mean(M1-M4), no-drift stream | verified |
| Drift benefit | 0.168 | confound summary | mean(M1-M4), drift stream | verified |
| Difference | 0.161 | confound summary | drift - no-drift | verified |
| Paired t / p | t=21.84, p<10^-37 | confound summary | paired t-test, 88 user means | verified |
| Cohen's d_z | 2.33 | confound summary | effect size | verified |
| Per-category drift benefit | 0.128 / 0.170 / 0.177 / 0.198 | confound summary | per-category (Table II) | verified |
| Dose-response MMD | slope 0.43, r=0.44 | confound + drift metrics | regression | verified |
| Dose-response KS | slope 0.47, r=0.37 | confound + drift metrics | regression | verified |
| Classifier-acc corr | rho=0.43 | confound + drift metrics | Spearman | verified |
| Baseline check | r=0.86, delta 0.0004-0.0068 | confound vs main M1 | correlation | verified |

## Results - Persistent-State RTRL (V-E)
| Claim | Value | Source | Computation | Status |
|---|---|---|---|---|
| Sweep comparisons | 330 valid | comparison.csv | n_with_eer per model | verified |
| M9 by learning rate | 0.321 / 0.258 / 0.210 / 0.160 | comparison.csv | mean_overall_eer per rate | verified |
| M2 baseline (sweep) | 0.151 | comparison.csv | mean_overall_eer, M2 lr=0.002 | verified |
| Paired t / p | t=4.91, p<10^-6 | M2/M9 0.002 summaries | paired t-test | verified |
| Ties / M2 wins / M9 wins | 51 / 170 / 109 | M2/M9 0.002 summaries | head-to-head | verified |

## Discussion - Touchalytics (cross-dataset)
| Claim | Value | Source | Computation | Status |
|---|---|---|---|---|
| Classifier acc all types | >0.88 | hmog_touchalytics_comparison.csv | min 0.8832 | verified |
| Per-category benefit (default) | +0.011 / +0.014 / -0.020 | touchalytics_adaptation_results.csv | per-drift-type M1-M4 | verified |
| Per-category significance | none significant | touchalytics_adaptation_results.csv | Wilcoxon per category | verified |
| Best-config cross-activity | 0.020, p=0.022 | touchalytics_sweep.log | sweep best line (n=40) | verified |
| RTRL=BPTT (touch) | abs(M2-M4)=0.0000 | touchalytics_adaptation_results.csv | per-category | verified |

## SUPERSEDED / DO NOT USE
- **stage4_rtrl_fix_20260414/** (Compute B): bug-fix validation run, 900 cross-both rows, M2 cross-both = 0.139 (NOT main factorial 0.172). Superseded by main factorial. NOT in paper.
- **9-row cv_summary partial** (Workstation C): stale; canonical is 50-row on Compute A.
- **n=876 M9 subset "crunch"**: computed in-chat from stage4_rtrl_fix file, never saved as analysis; conclusion was "inconclusive." Superseded by the LR sweep (comparison.csv). NOT in paper.

## Disposition note

Superseded, partial, failed, or unfavorable results are part of the scientific provenance. “Superseded” does not mean “safe to delete.” Archival or removal requires an approved disposition record identifying the retained location and replacement evidence.

# Paper-to-Code and Artifact Map

## Status terms

- **Inspected:** source was read as text in the current checkout.
- **Candidate:** a plausible analysis or rendering path that has not been executed and compared with the manuscript.
- **Manuscript-reported:** stated in the designated manuscript source; supporting artifacts are unavailable or unverified here.
- **Historical-ledger reference:** named by the pre-existing provenance ledger, not established as the final repaired artifact by filename alone.
- **Prior-recorded verification:** a check documented by a retained record; integrated records establish only their stated scope.
- **Reanalysis verified:** a named artifact-based calculation has a retained successful execution and output comparison, with exceptions stated. Not full retraining.
- **Pending:** required implementation or artifact evidence has not been identified.

The paper and manuscript source are not required by the saved-output commands and are intentionally kept separate from the code update. Manuscript labels below identify scientific roles, not files that a reviewer must copy into the repository. Preserve the stated scope of historical records and new executions.

## Current artifact and execution index

The current-route update is based on `a017f58ef14a31d50b30c5b9e37fd8c8a1281548` and its October 5 source-snapshot execution. Existing experimental definitions and manuscript labels are retained; this is not a new manuscript/PDF audit, fresh dependency installation or raw-data experiment.

For compactness, the tables below use these repository-relative package keys:

| Key | Directory and evidence |
|---|---|
| **F** | [`artifacts/reference/m7_repaired_20260901/`](../artifacts/reference/m7_repaired_20260901/), including accepted `results_stage4_merged_summary_m7_repaired_20260901.csv`, merge/acceptance reports, and `m7_m8_statistical_closure/`. |
| **C** | [`artifacts/reference/matched_stream_m9/`](../artifacts/reference/matched_stream_m9/), including `results_confound/`, five summaries and `comparison.csv` in `results_m9_lr_sweep/`, saved M9 figure, and `source_snapshot/`. |
| **D** | [`artifacts/reference/drift_touchalytics_diagnostics/`](../artifacts/reference/drift_touchalytics_diagnostics/), including `results_stage4b_exp1/`, `touchalytics_data/`, and `source_snapshot/`. |
| **T** | [`artifacts/reference/touchalytics_adaptation/`](../artifacts/reference/touchalytics_adaptation/), including `results_touchalytics/`, `results_touchalytics_raw/`, full sweep/run logs, and `source_snapshot/`. |
| **E** | [October reanalysis evidence](evidence/reproduction_20261004/README.md), including environment, execution, and comparison records. |

The older M7–M8, matched-stream and M9 sweep execution records remain scoped to their recorded environment. New executable routes now include the two descriptive tables, M9 user-level inference and separate Touchalytics analyses. The five relocated publication variants also completed. In the newer environment, two M7 statistical CSVs have last-digit differences and are not byte-identical; its user-value CSVs remain exact. See [the central guide](REPRODUCING.md) for actual versions, comparison outcomes, limits and commands.

## Experimental definitions

| Paper concept | Manuscript location | Defining source | What inspection establishes | Status |
|---|---|---|---|---|
| Sessions 1-3 for training and session 4 for streaming | `fig:setup`; Dataset; Training Protocol | `run_stage4.py`, `data_loader.py`, `run_experiment.py` | The primary driver selects the first three qualifying training sessions and condition-specific fourth streaming session | Inspected |
| Six conditions and 36 ordered pairs | `fig:setup`; `fig:matrix` | `run_stage4.py`; `run_streaming.get_pairs` | Six activity-posture conditions and their Cartesian product | Inspected |
| Four drift categories | Problem Definition; `fig:matrix` | `run_streaming.classify_pair` | Temporal, cross-posture, cross-activity, and `cross-both` assignments | Inspected |
| Balanced stream and deterministic interleaving | Experimental Design; Training Protocol | `run_stage4.run_stage4_experiment` | Excess impostor windows are downsampled; class order is preserved before seeded interleaving | Inspected |
| Deterministic impostor selection | Experimental Design | `run_streaming.get_impostors`; caller in `run_stage4.py` | Three identities are sampled after excluding the current target | Inspected |
| Prediction before one window-boundary update | Window-Based Incremental Adaptation | M2, M4, M7, and M8 branches in `run_stage4.py` | Each factorial adapter records a score before updating | Inspected |
| M3 legitimate-only exception | `tab:models`; Window-Based Incremental Adaptation | M3 initialization and `rtrl_adapt_legit` branch in `run_stage4.py` | M3 scores every window but updates only when the ground-truth label is legitimate | Inspected |
| Model identities | `tab:models`; `subsubsec:m9` | `run_stage4.ALL_MODELS`, `MODEL_NAMES`, and streaming branches | Implements M1-M9 roles used by the documentation | Inspected |
| EER and temporal summaries | Evaluation Metrics | `run_experiment.compute_eer`; summary logic in `run_stage4.py`; rolling logic in `run_streaming.py` | Candidate implementation of threshold-grid EER, initial/end EER, and rolling EER | Inspected; numerical correspondence unexecuted |

## Manuscript-rendered diagrams and table

### `fig:setup` — Experimental data setup

- **Scientific definition:** session selection in `run_stage4.py`.
- **Rendering source:** a TikZ session diagram and LaTeX table defined directly in the supplied manuscript source.
- **Repository renderer:** none identified or required.
- **Status:** code definition inspected; manuscript rendering described from the supplied source.

### `fig:matrix` — Condition-pair matrix

- **Scientific definition:** `run_streaming.get_pairs` creates the 36 ordered pairs and `run_streaming.classify_pair` assigns categories.
- **Rendering source:** a manually enumerated TikZ 6-by-6 matrix in the supplied manuscript source.
- **Repository renderer:** none identified.
- **Pending check:** compare every manually shaded manuscript cell with the programmatic category assignment.

### `tab:models` — Experimental Models and Their Roles

- **Scientific definition:** the model registry and streaming initialization/update branches in `run_stage4.py`.
- **Rendering source:** a manually assembled LaTeX table in the supplied manuscript.
- **Artifact dependency:** none.
- **Status:** manuscript and primary driver agree on M1-M8; M9 is separate.

## RQ0 — Drift and degradation without adaptation

| Result or figure | Condition-defining code | Result-computing code | Input artifact | Rendering code or source | Status |
|---|---|---|---|---|---|
| Feature-level KS and source-classifier results | `exp1_50_users.txt`; category logic in `run_drift_detection.py` | `run_drift_detection.py` | HMOG | None for numerical results | Inspected; not executed |
| `fig:drift_detection` | Same | Same | D: saved drift-detection summary | `analysis/publication_figures/generate_drift_detection_figure.py` | Direct execution passed; reconstructed presentation, not pixel-identical historical source |
| Frozen M1/M5/M6 means | Model paths in `run_stage4.py` | Full set of final aggregations still to be linked; M6 descriptive views occur in the verified M7 script | F: accepted repaired summary | Manuscript prose | Input present; do not extend the verified M6/M7/M8 route to untested M1/M5 summaries |

The 50-combination drift subset contains temporal, cross-activity, and cross-both cases and excludes cross-posture cases. Its selected-subset results do not establish full-cohort category distributions.

## RQ1 — Distribution shift versus insufficient training data

| Result or figure | Condition-defining code | Result-computing code | Input/output artifact | Rendering code | Status |
|---|---|---|---|---|---|
| Held-out earlier-session and session-4 EER | D: `source_snapshot/exp1_50_users.txt` | `run_stage4b_exp1.py` | D: `results_stage4b_exp1/results_stage4b_cv_summary.csv` | None | Input identity/coverage retained; saved-output summaries exercised by renderer; no raw-data retraining |
| Drift gap | Same | `run_stage4b_exp1.py` | Same summary | None | Stored values summarized by the tested drift-gap renderer; original experiment not rerun |
| `fig:drift_gap` | Same | Drift gap above | Same summary | `analysis/publication_figures/generate_drift_gap_figure.py` | Direct execution passed; reconstructed presentation |
| KS/classifier association with drift gap | Same | Candidate route in `run_drift_detection.py`; final saved analysis unverified | Both drift summaries | Manuscript prose | Pending verification |

The selected 50 combinations come from 42 users, were stratified using frozen-model difficulty, and do not include cross-posture pairs.

## RQ2 — Window-based adaptation

### Main factorial

| Component | Source | Role | Status |
|---|---|---|---|
| Experiment driver | `run_stage4.py` | Trains and streams M1-M9 | Inspected |
| VanillaRNN gradients | `model.py` | M1-M4 and M9 implementation | Inspected |
| GRU RTRL and transfer | `gru_model.py` | M7 implementation and PyTorch weight transfer | Inspected |
| PyTorch models | `pytorch_models.py` | Offline LSTM/GRU and M8 BPTT | Inspected |
| Factorial summary | F: `results_stage4_merged_summary_m7_repaired_20260901.csv` | Accepted repaired attempted-output summary, SHA-256 in provenance | Present; used in verified M7–M8 reanalysis |
| Factorial detail | Historical `results_stage4_merged_detail_with_m7.csv` | Per-window output is separate from repaired summary | Full-cohort repaired detail not packaged; two-chunk acceptance record is present in F |
| Generic chunk merge | `merge_results.py` | Merges and deduplicates chunk CSVs | Inspected; not established as final repair merge route |

### `tab:percategory`

- **Reported statistic:** configuration-weighted M1 and M2 EER by drift category.
- **Condition definition:** `run_streaming.classify_pair`.
- **Expected input:** F: accepted repaired summary; the metric-specific completeness rule must be explicit.
- **Computing implementation:** `analysis/reproduce_summary_tables.py`; differences are calculated before rounding.
- **Rendering source:** manually assembled LaTeX table in the manuscript.
- **Status:** new table code executed from the source snapshot; all displayed cells match the checked rounding, not a claim to recover historical table-generation source.

### `fig:forgetting`

- **Definition:** M2 end EER minus M1 end EER with a `+0.05` forgetting threshold.
- **Expected input:** final repaired factorial summary.
- **Renderer:** `analysis/publication_figures/generate_forgetting_histogram.py` (reconstructed publication variant).
- **Expected output:** `figures/forgetting_histogram_m2.png` and `.pdf` relative to the command's working directory.
- **Status:** direct execution passed; prior captured numerical-payload match retained. Pixel-identical manuscript reproduction is not established.

### Recovery metrics and supporting curve

Initial, end, and rolling EER are recorded by the main pipeline. `generate_recovery_curve.py` is retained as supporting or historical analysis and writes `figures/recovery_curve.png` and `.pdf`. The designated manuscript contains no included recovery-curve figure, so this script is not a reproduction route for an included figure.

### M3 ablation

M3 scores every stream window and updates only after legitimate-user windows. The M2-M3 comparison therefore examines the contribution of correctly labeled impostor updates under the oracle-supervised protocol. F supplies the repaired summary; the final M2-M3 aggregation remains a distinct route to link, not part of the verified M7–M8 inference.

## `subsubsec:confound` — Matched-stream control

| Result or figure | Experimental definition | Result computation | Artifact | Rendering | Status |
|---|---|---|---|---|---|
| Matched control/drift streams | `run_confound.py` | `run_confound.py` | C: `results_confound/results_confound_summary.csv` and `results_confound_failed.csv` | None | Saved inputs present and structurally inspected; original experiment not rerun |
| Primary paired and mixed-effects results | Same | E: `records/matched_stream/analyze_confound.py` (tested snapshot) | Same plus F summary staged as `results_stage4_merged_summary.csv` | Saved text report | Reanalysis verified at printed precision except initial coverage-line ordering; warnings preserved |
| `fig:paired_bar` | Same | User/category aggregation | C: confound summary | `analysis/publication_figures/generate_confound_figures.py` | Direct execution passed; prior numerical-payload match, not manuscript pixel equality |
| `tab:confound` | Same | `analysis/reproduce_summary_tables.py` | C: confound summary | Locally generated CSV/stdout | New code executed; equal users within category and equal category weights overall |
| `fig:dose_response` | Same | MMD/KS and benefit differences | C: confound summary | `analysis/publication_figures/generate_confound_figures.py` | Direct execution passed; descriptive nested observations remain qualified |

This experiment compares M1 and M4 on separate matched earlier-session and later-session streams, distinct from the architecture-gradient comparison. The imported records contain 9,396 summary keys and 108 disjoint failure keys (9,504 combined); the retained report uses 88 target users. The one-user-one-value primary result must remain distinct from user-by-category diagnostics. The separate table CLI and publication-rendering routes above must not be conflated with report reproduction; none requires storing the manuscript in the repository.

The retained report/source contain documented description-versus-implementation distinctions for the Holm family and mixed-effects random slope. Reanalysis preserves rather than resolves those distinctions and the boundary/Hessian warnings. See C's packaging notes and E's evidence README.

## RQ3 — RTRL, BPTT, and computational cost

### VanillaRNN comparison

- **Models:** M2 versus M4.
- **Shared initialization:** clones of the same offline-trained VanillaRNN in `run_stage4.py`.
- **Difference:** forward sensitivities versus backward propagation within the same window.
- **Expected input:** final repaired factorial summary.
- **Status:** implementation inspected; reported result not recomputed.

### GRU comparison and M7 repair

- **Models:** M7 versus M8, with M6 as frozen GRU control.
- **Current-code initialization:** M7 receives transferred trained-GRU weights; M8 deep-copies the trained GRU.
- **Integrated repair evidence:** F contains merge provenance and prior summary/two-chunk detail acceptance reports. They record 16 repaired users, exact M6/M8 control agreement, unchanged failure keys, and replacement of only the designated M7 values.
- **Boundary:** original audit/launch/input records and full-cohort repaired-detail assembly are not all in the package. Presence of the acceptance reports does not establish every historical link or full retraining.

### User-level M7-M8 inference

[The recovered script](../analysis/reproduce_m7_m8_statistics.py) and F's four accepted statistical CSVs/report are present. E records the completed 89-user execution with 100,000 bootstrap resamples, paired tests, equal-category sensitivity, category-specific results, and exact four-CSV agreement. The report matches after normalizing only the candidate-input path. This route is no longer unidentified or unexecuted.

The recovered rule retains rows with complete numeric M6/M7/M8 overall-EER triples and rejects partial triples; on the accepted 9,558-row hash it retains 9,216 configurations. Its primary inference averages within users and gives users equal weight, separately from the configuration-weighted descriptive view. Do not use its coverage assertion as a newly established all-model/all-metric filter or treat the small detected difference as formal equivalence.

### Complexity

The manuscript provides a theoretical asymptotic comparison. Relevant implementations are `model.py`, `gru_model.py`, and the PyTorch BPTT paths. It is not a measured smartphone latency, memory, or energy result.

## `subsubsec:m9` and `fig:m9_sweep`

| Component | Path | Status |
|---|---|---|
| Persistent-state implementation | M9 branch in `run_stage4.py`; persistent methods in `model.py` | Inspected |
| Sweep driver | `run_m9_lr_sweep.py` | Inspected; not executed |
| Configuration summaries | C: five `results_m9_lr_sweep/*_summary.csv` files | Present; 360 rows each, 330 finite/nonempty model EERs each; detail/curve outputs not included |
| Configuration-weighted comparison | C: `results_m9_lr_sweep/comparison.csv`; `analysis/reproduce_m9_comparison.py` | Reanalysis verified: byte-identical comparison CSV; E retains execution and comparison |
| `fig:m9_sweep` renderer | `analysis/publication_figures/generate_m9_sweep_figure.py` | Direct execution passed; reconstructed image, not exact historical PNG reproduction |
| Persistent-state numerical checks | `tests/test_rtrl_persistent.py`; `tests/test_rtrl_persistent_gradcheck.py` | Source-snapshot tests passed; fixed-parameter toy sequences only |
| User-level M9 inference | `analysis/reproduce_m9_user_statistics.py` | New code executed; explicitly t-based CI, paired user means and exploratory/post-selection status |

M9 is an exploratory validation-user analysis and is not part of the full-cohort architecture-gradient comparison.

## Touchalytics exploratory analysis

| Result | Candidate code | Candidate artifact | Status |
|---|---|---|---|
| Drift diagnostics | D: `source_snapshot/touchalytics_drift_check.py` | D: `touchalytics_data/touchalytics_drift_results.csv` | 68 rows / 41 users; historical identity reconciled, new reanalysis pending |
| HMOG/Touchalytics table | D: `source_snapshot/touchalytics_hmog_comparison.py` | D: `touchalytics_data/hmog_touchalytics_comparison.csv` | Present but HMOG cells blank; not a completed two-dataset comparison |
| Default feature-matrix adaptation | Producer mismatch retained; new `analysis/reproduce_touchalytics_statistics.py` consumes saved results | T: default 180-row CSV | Artifact-based reanalysis executed; exact historical producer not established |
| Raw-event adaptation | T: `source_snapshot/run_touchalytics_raw_adaptation.py` | T: `results_touchalytics_raw/touchalytics_raw_adaptation_results.csv` | Separate 204-row historical result present; not interchangeable with default |
| Raw-event analysis | `analysis/reproduce_touchalytics_statistics.py` | T: separate raw-event 204-row CSV | Reanalysis executed independently of default results; unadjusted paired tests |
| Sequential sweep | `analysis/reproduce_touchalytics_statistics.py` reports the saved log | T: unchanged full sweep log | All four round tables/final setting reported; selected-sweep p-value not recomputed |

The release must preserve the separate default and raw-event outcomes and the post-selection qualification of the selected sweep result. Nonsignificant VanillaRNN comparisons must not be generalized to all model/category comparisons; the new reanalysis also exposes unadjusted raw-event temporal GRU results. The full log records the sequentially selected W=50, online learning rate 0.002, `raw_dyn6`, H=32 configuration; that is a retained log observation, not independent confirmation. The earlier imports did not rerun those tests; the new artifact-based CLI has now executed in the environment stated in the central guide. No alternate-name CSV was created to conceal the producer mismatch; see T's packaging notes.

## Artifact validation record

For each supplied artifact, integrate its retained filename, size, hash, schema, primary key, duplicate count, attempted/failed/complete representation, producing source version, command, neutral machine identifier, merge or repair history, review checkpoint, and consuming manuscript items. Preserve what each prior check established: a file inventory or matching hash must not be promoted into evidence for an untested statistical result. A filename containing `merged`, `with_m7`, or `final` is not sufficient evidence of canonical status.

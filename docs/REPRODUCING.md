# Reproducing the Analyses and Experiments

## Scope and tested source

The package distinguishes **Level A** (numerical/unit and packaging checks), **Level B** (reanalysis of saved experiment outputs), and **Level C** (raw-data training and streaming). Passing one level does not establish the others.

The 2026-10-05 workflow test used a fresh GitHub source archive matching commit `a017f58ef14a31d50b30c5b9e37fd8c8a1281548`, tree `c49e9820f21570affe7c950826451ff97c77eeac`. All 247 file contents and executable modes were included in that tree check. Four numerical-test programs, six analysis commands, and five relocated publication-rendering commands completed in isolated working directories. The uploaded source archive and all tracked source/reference files remained unchanged.

This was a **fresh source snapshot in an existing Linux environment**, not a fresh package installation or a Git-history review. No source dataset was loaded, and no HMOG/Touchalytics experiment was retrained. The GRU test includes its existing small synthetic training comparison. Logs and inspection reports from this workflow are retained separately, not added as manuscript/PDF-check material to the repository.

## Environments: distinguish execution from installation

| Record | Versions and scope |
|---|---|
| Historical model training | Not established by this package; neither requirements specification is asserted to be its lock file. |
| Recorded October 3–4 reanalysis | CPython 3.13.2; NumPy 2.2.6; Pandas 2.2.3; SciPy 1.15.3. Matched-stream/figure runs additionally used Matplotlib 3.10.3, Statsmodels 0.14.5 and Patsy 1.0.1. |
| October 5 source-snapshot execution | CPython 3.13.5, Linux x86_64; NumPy 2.3.5; Pandas 2.2.3; SciPy 1.17.0; Matplotlib 3.10.8; PyTorch 2.10.0+cpu; Statsmodels 0.14.6; Patsy 1.0.2. Existing installations were used; nothing was installed or upgraded for this check. |
| October 20 CPU reviewer environment | CPython 3.13.2, Linux x86_64 (glibc 2.35); NumPy 2.2.6; Pandas 2.2.3; SciPy 1.15.3; PyTorch 2.7.0+cpu; scikit-learn 1.6.1; Matplotlib 3.10.3; Seaborn 0.13.2; Statsmodels 0.14.5; Patsy 1.0.1; python-dotenv 1.2.1; Joblib 1.5.1; tqdm 4.67.1. Fresh environment and focused execution check; details below. |

For the newly assembled CPU reviewer environment, follow the
[Python environment and installation section in the root README](../README.md#python-environment-and-installation).
It specifies CPython **3.13.2**, Linux x86_64 (observed glibc 2.35), and
PyTorch **2.7.0+cpu**. Install PyTorch from its official CPU index first, then
install the root `requirements.txt` from PyPI. The root file now records the
**35 installed versions** exported from that environment, including pip and
pip-chill; it replaces the provisional root dependency list.

The separate [October 3–4 reanalysis version list](../requirements/reanalysis-python313.txt)
and [installation records](evidence/reproduction_20261004/records/environment/)
remain unchanged. Neither those records nor the new requirements file establish
the original training environment. Do not combine the two version lists or
modify a preserved working environment to modernize its dependencies.

### October 20 CPU environment compatibility check

The environment was installed in a fresh `venv` using 12 explicit project
package pins, with CPU PyTorch obtained separately from its official index.
The installed-package consistency check (`pip check`) and the 12 library
imports passed. The full 35-package version record was exported after that
installation. The newly packaged full-pin installation sequence has not yet
been replayed in a second fresh environment; it is not a wheel-hash lock or a
claim of cross-platform installation support.

The maintainer-run check used organization-repository commit
`1ad1fcc846a5ee1e709a22308b2a6b6d5de923f5` (tree
`4e1552e88d04e0a44c95b81ce3d701cde34fcd54`). Its completion summary reports nine
completed commands, no missing required outputs, and unchanged tracked source
and staged inputs. The command set was:

| Scope | Commands |
|---|---|
| Numerical tests | `test_gru_forward.py`, `test_gru_rtrl.py`, `tests/test_rtrl_persistent.py`, `tests/test_rtrl_persistent_gradcheck.py` |
| Self-contained analyses | `analysis/reproduce_summary_tables.py`, `analysis/reproduce_m9_user_statistics.py`, `analysis/reproduce_touchalytics_statistics.py` |
| Matched-stream analysis | `analyze_confound.py`, with copies of the reference CSVs staged in a separate working directory |
| Figure execution | `analysis/publication_figures/generate_m9_sweep_figure.py`, with outputs in a new working directory |

The reviewed terminal excerpt contains the end of the matched-stream report,
the M9 figure output and the final nine-command completion summary; it is not a
complete transcript of all nine individual command logs. Full per-command logs
remain in the maintainer's private check directory. No private paths or logs are
added here.

This was a scoped compatibility check, not an exact comparison of scientific
outputs or figures. M7–M8 statistical reanalysis and the other four publication
renderer entry points were not included in this nine-command run. The
matched-stream report still showed boundary and non-positive-definite Hessian
warnings. The four numerical tests use synthetic inputs; the GRU test includes
its small existing offline-training comparison. No HMOG or Touchalytics
raw-dataset training experiments were rerun.

Use `.venv/bin/python` instead of `python` in subsequent commands, or activate
the new environment first. Keep the isolated-output precautions below. The
October 20 record does not replace or extend the earlier historical checks or
their equality criteria.

### Output comparisons in the October 5 environment

| Route | Observed outcome | What remains limited |
|---|---|---|
| Two descriptive tables | 27 displayed cells agreed with the previously checked rounded values. | New table code, not recovered historical table-generation source. |
| M9 user-level statistics | 330 matched successful keys, 10 users; agreement at previously reported precision. | Explicit t-based interval and unadjusted post-selection inference, not independent confirmation or historical-source identification. |
| Touchalytics statistics | Default/raw inputs remained separate; 24 unadjusted paired comparisons and all four sweep-round excerpts produced. | Selected-sweep per-user data absent; its p-value is only reported from the log. |
| M7–M8 | User-value and user/category-value CSVs byte-identical; report matched after input-path normalization. | Overall/category statistical CSVs were **not** byte-identical: 12 last-digit changes, maximum absolute difference `3.47e-18`. Strict four-CSV comparison fails in this environment. |
| M9 sweep | Comparison CSV byte-identical. | Configuration weighting only, separate from user inference. |
| Matched-stream report | Matched except for the same four initial coverage-line ordering differences. | Strict byte equality remains false; boundary/Hessian warnings preserved. |
| Publication-rendering commands | Five direct commands produced all ten expected PNG/PDF pairs (seven included image files, three supporting plots). | No new pixel-equality or historical PDF match is claimed. Prior reconstructed-rendering review remains the applicable comparison. |

The [October 3–4 evidence](evidence/reproduction_20261004/README.md) still establishes four exact M7–M8 CSV matches in its earlier recorded environment. Do not recast the October 5 last-digit differences as an exact match, replace references, or relax the original criterion. This compatibility test does not reopen that completed historical comparison.

## Safety and data access

Raw HMOG sensor data, raw Touchalytics events and feature matrices are not needed for Levels A/B and are not supplied. Derived reference CSVs include participant codes and require a separate redistribution review. The paper, manuscript source, PDF/source-check reports and manuscript-page screenshots are not required by the commands below.

Run from a disposable source copy. For scripts with fixed relative output names, use a new working folder and stage unchanged copies of their inputs. Never run such scripts inside `artifacts/reference/` or a preserved results directory. In particular:

- `analysis/reproduce_m9_comparison.py` overwrites its relative `comparison.csv` and can skip missing inputs; all five inputs must be staged and the result compared as below.
- Matched-stream report and figure programs write below the current `results_confound/`; positional-input figure programs write below the current `figures/`.
- Do **not** use `smoke_test.sh` as a dataset-independent check: it deletes two confound result files. `run_confound_parallel.sh` starts 12 jobs and moves/merges output files. Neither is a reviewer quick start.

For Level C only, configure dataset roots locally via `HMOG_ROOT` or the Touchalytics script's documented `TOUCHALYTICS_DATA`/positional path. Do not commit credentials, private connection details or raw participant data.

## Level A: numerical tests

From the source-copy root in an environment containing NumPy and PyTorch:

```bash
set -euo pipefail
python -B test_gru_forward.py
python -B test_gru_rtrl.py
python -B tests/test_rtrl_persistent.py
python -B tests/test_rtrl_persistent_gradcheck.py
```

All four programs passed their printed/asserted numerical checks in the October 5 environment. Gate 4 in `test_gru_rtrl.py` checks transferred-weight forward agreement and five bias-gradient arrays; do not describe it as checking every H=32 weight gradient. The smaller finite-difference gates cover their listed parameter arrays. Persistent-state tests hold parameters fixed while checking state threading and derivatives; they do not validate every evolving online training trajectory.

The integration changes make the GRU programs return failure status when their Boolean checks fail, and explicitly extract the scalar loss in the persistent-gradient finite-difference helper. Core models and tolerance thresholds are unchanged. The three revised programs also passed with deprecation warnings treated as errors; their stdout matched the original successful runs. Controlled false-result tests verified the revised failure statuses.

`tests/smoke_test_m9.py` requires HMOG and runs an experiment. It was deliberately not executed as a unit test. Do not use indiscriminate test discovery to start dataset-backed checks.

## Level B: reference integrity

Use only the manifest describing the distributed copy and the indicated working directory. These checks verify file identity, not scientific correctness.

| Manifest | Working directory |
|---|---|
| `MANIFEST.sha256` | `artifacts/reference/m7_repaired_20260901/` |
| `MANIFEST.sha256` | Its `m7_m8_statistical_closure/` subdirectory |
| `COLLECTION_MANIFEST.sha256` | `artifacts/reference/matched_stream_m9/` |
| `PACKAGED_MANIFEST.sha256` | `artifacts/reference/drift_touchalytics_diagnostics/` |
| `PACKAGED_MANIFEST.sha256` | `artifacts/reference/touchalytics_adaptation/` |
| `docs/evidence/reproduction_20261004/PACKAGED_MANIFEST.sha256` | Repository root |
| `docs/evidence/publication_rendering_20261004/PACKAGED_MANIFEST.sha256` | Repository root |

All 163 entries across these seven distributed-layout manifests passed in the tested snapshot. On Linux use `sha256sum -c MANIFEST_NAME` from the stated directory. Original pre-sanitization manifests are historical records; they intentionally need not validate sanitized copies. Do not regenerate them to conceal documented differences.

### The three self-contained analysis commands

From the source-copy root:

```bash
python -I -B analysis/reproduce_summary_tables.py
python -I -B analysis/reproduce_m9_user_statistics.py
python -I -B analysis/reproduce_touchalytics_statistics.py
```

These commands need only existing reference files, not the paper, raw data, training modules or a server connection. Each validates its fixed SHA-256 inputs and prints to stdout. `reproduce_summary_tables.py` uses only the standard library; the other two also need NumPy/SciPy. The M9 script requires SciPy's paired-test confidence-interval interface.

Use `--output-dir /absolute/path/to/NEW-directory` to save generated CSVs locally. The parent must already exist. All three rejected existing output folders, protected artifact destinations, missing inputs and altered input hashes in the exercised negative cases, without overwriting the sentinel files. Existing-result protection is not a full adversarial filesystem/race-condition security audit.

Table calculations preserve configuration weights for the factorial table and equal user weights within category for the matched-stream table; its overall row weights categories equally. Differences are computed before rounding. The M9 comparison weights users equally after within-user configuration averaging and remains exploratory/post-selection. Touchalytics averages seeds within user/category, keeps the two experiments separate, and retains unadjusted tests. Nonsignificance of a VanillaRNN comparison must not be generalized to every GRU comparison. Near-zero skipped tests are not equivalence tests.

### Isolated staging for the legacy analysis entry points

These are Bash/Linux commands. Start at the root of the disposable copy and use an already prepared interpreter:

```bash
set -euo pipefail
umask 077
REPO="$(pwd -P)"
PY="$(python -I -c 'import sys; print(sys.executable)')"
WORK="$(mktemp -d)"
M7="$REPO/artifacts/reference/m7_repaired_20260901"
CM="$REPO/artifacts/reference/matched_stream_m9"
EV="$REPO/docs/evidence/reproduction_20261004"
printf 'Workspace: %s\n' "$WORK"
```

#### M7–M8 user-level statistics

```bash
mkdir "$WORK/m7"
cp "$REPO/analysis/reproduce_m7_m8_statistics.py" "$WORK/m7/"
cp "$M7/results_stage4_merged_summary_m7_repaired_20260901.csv" "$WORK/m7/input.csv"
"$PY" -I -B -u "$WORK/m7/reproduce_m7_m8_statistics.py"   --input "$WORK/m7/input.csv" --output-dir "$WORK/m7/results"   > "$WORK/m7/execution.log" 2>&1
```

The input is the accepted repaired summary (`72170629...` SHA-256 as checked by the program). The recovered rule rejects partial M6/M7/M8 triples and selects complete triples; it is not automatically the inclusion rule for all other models/metrics. The existing bootstrap and sign-flip calculations run; this is not retraining.

Check each of the four CSVs against its reference with `cmp`. A nonzero comparison exit is a mismatch to preserve, not an instruction to overwrite anything:

```bash
for name in m7_m8_overall_statistics.csv m7_m8_by_category.csv             m7_m8_user_level_values.csv m7_m8_user_category_values.csv; do
  if cmp "$M7/m7_m8_statistical_closure/$name" "$WORK/m7/results/$name"; then
    printf 'EXACT: %s\n' "$name"
  else
    printf 'REVIEW REQUIRED: %s differs; preserve both copies\n' "$name"
  fi
done
```

The report has an input path; its historical criterion normalizes only the value of the single `Candidate file:` line in comparison memory. Nothing else is normalized. A new path also changes the report hash and hence the generated manifest. The strict four-CSV criterion is known not to pass in the newer October 5 environment described above.

#### Matched-stream report

```bash
mkdir "$WORK/matched_stream"
mkdir "$WORK/matched_stream/results_confound"
cp "$EV/records/matched_stream/analyze_confound.py" "$WORK/matched_stream/"
cp "$CM/results_confound/results_confound_summary.csv"    "$CM/results_confound/results_confound_failed.csv"    "$WORK/matched_stream/results_confound/"
cp "$M7/results_stage4_merged_summary_m7_repaired_20260901.csv"    "$WORK/matched_stream/results_stage4_merged_summary.csv"
(cd "$WORK/matched_stream" &&   MPLCONFIGDIR="$WORK/matched_stream/matplotlib_config"   "$PY" -I -B -u analyze_confound.py > execution.log 2>&1)
```

Use the tested snapshot, not a guessed current/historical producer. The factorial input alias is a staged copy, never a rename of the canonical file. Omitting it skips the final baseline section. The program emits reports and diagnostic figures. Only the four initial coverage-row ordering differences are covered by the historical report exception; do not sort the whole report. All other report bytes matched in the October 5 test as well.

The mixed-effects boundary/non-positive-definite-Hessian warnings remain. Historical wording about the Holm family and random intercept does not fully describe the retained implementation: separate three-test adjustments and a stream-type random slope are present. Reproducing its output does not validate every historical explanation or inference assumption; see the existing [packaging notes](../artifacts/reference/matched_stream_m9/PACKAGING_NOTES.md).

#### M9 configuration-weighted comparison

```bash
mkdir "$WORK/m9"
mkdir "$WORK/m9/results_m9_lr_sweep"
cp "$REPO/analysis/reproduce_m9_comparison.py" "$WORK/m9/"
cp "$CM/source_snapshot/validation_users.txt" "$WORK/m9/"
for tag in M2_lr_0.002 M9_lr_0.0001 M9_lr_0.0005 M9_lr_0.001 M9_lr_0.002; do
  test -f "$CM/results_m9_lr_sweep/${tag}_summary.csv"
  cp "$CM/results_m9_lr_sweep/${tag}_summary.csv" "$WORK/m9/results_m9_lr_sweep/"
done
(cd "$WORK/m9" && "$PY" -I -B -u reproduce_m9_comparison.py > execution.log 2>&1)
cmp "$CM/results_m9_lr_sweep/comparison.csv" "$WORK/m9/results_m9_lr_sweep/comparison.csv"
```

All five configurations are required despite the helper's historical skip behavior. Each has 360 attempted and 330 usable configuration rows. The CSV exact comparison passed in both recorded environments. This does not replace the distinct M9 user-level analysis above.

### Figures

Use the direct isolated commands in the [publication-rendering guide](../analysis/publication_figures/README.md), not the older root-level renderers by accident. All five relocated scripts completed from the fresh snapshot. They recompute the summaries/bootstrap needed by the plots but do not retrain models. Their prior captured numerical payloads matched their original counterparts; reconstructed output images remain different from manuscript references. No pixel criterion was relaxed, and no manuscript image was replaced.

`generate_recovery_curve.py` is supporting/historical analysis, not a route for an included manuscript image. It needs per-window detail data not supplied here and was not run.

## Remaining work and stop rules

The known artifact collection, previously accepted scientific comparisons, table rounding review and figure-presentation reconciliation are not reopened by packaging work. The new snapshot test establishes execution only for the named commands and checks.

| Area | Remaining boundary |
|---|---|
| Installation | Fresh installation from an exported requirements list was not exercised; the snapshot test used existing packages. Do not advertise a tested installer or universal cross-platform identity. |
| Statistical/figure identity | Preserve the documented M7 last-digit and figure pixel differences; exact historical outputs remain environment-specific. |
| Additional paper claims | This walkthrough is not a complete re-audit of every prose statistic or the 6-by-6 manuscript diagram. Full repaired-detail assembly and any additional summary route not covered by the named commands remain separate. |
| Full experiments | Raw datasets, original training environment and the default Touchalytics producer mismatch are not resolved by saved-output reanalysis. No full rerun without an explicit budget/decision. |
| Release | Software licensing, derived-output rights, Git-history/privacy review, final naming/branch cleanup, version freeze and organization handoff still require completion. |

Fix concrete execution, correctness, safety or instruction problems; defer optional improvements. A missing record is not proof of failure and must not be relabeled as success. Keep manuscript/source/PDF checks outside the code update, and do not silently delete earlier approved evidence.

## Level C: full HMOG experiments

### Main factorial interface

`run_stage4.py` defines these arguments:

```text
--users-file PATH
--pairs {all,cross-posture,cross-activity,cross-both,temporal}
--models COMMA_SEPARATED_MODEL_IDS
--seeds COMMA_SEPARATED_SEEDS
--chunk INTEGER
--num-chunks INTEGER
--summary PATH
--detail PATH
```

The complete factorial is a large experiment, not a quick check. No full-run command is designated until the environment, data layout, compute budget, output isolation, and resume behavior have been validated. In particular, exception rows are written to the summary, which also acts as the resume marker; incomplete-row behavior must be understood before relying on automatic resume.

### Drift-verification subset

`exp1_50_users.txt` records the selected target/condition combinations. A single model-based drift-gap run can be expressed as:

```bash
cd /path/to/reproduction-checkout
python run_stage4b_exp1.py \
  --user TARGET_ID \
  --train-condition ACTIVITY+POSTURE \
  --stream-condition ACTIVITY+POSTURE \
  --seed 42 \
  --output-dir /path/to/reproduction-checkout/new-results/drift-verification
```

Feature-level drift for one combination:

```bash
cd /path/to/reproduction-checkout
python run_drift_detection.py \
  --user TARGET_ID \
  --train-condition ACTIVITY+POSTURE \
  --stream-condition ACTIVITY+POSTURE \
  --output-dir /path/to/reproduction-checkout/new-results/drift-verification
```

Feature-level drift for all entries in `exp1_50_users.txt`:

```bash
cd /path/to/reproduction-checkout
python run_drift_detection.py \
  --all \
  --output-dir /path/to/reproduction-checkout/new-results/drift-verification
```

These are unexecuted templates. The selected subset contains temporal, cross-activity, and cross-both combinations and does not represent the full factorial population.

### Matched-stream experiment

`run_confound.py` defines user-list, seed, pair-category, chunk, and holdout-mode arguments. Its default holdout mode is `contiguous`, but it writes to the fixed `results_confound/` directory. Run it only from a disposable checkout:

```bash
cd /path/to/reproduction-checkout
python run_confound.py \
  --users-file test_users.txt \
  --seeds 42,123,456 \
  --pairs all \
  --holdout-mode contiguous
```

### Exploratory M9 sweep

From a disposable checkout with HMOG configured:

```bash
cd /path/to/reproduction-checkout
python run_m9_lr_sweep.py
```

This uses `validation_users.txt`, the cross-both pairs, and the five configured M2/M9 learning-rate combinations, and writes under `results_m9_lr_sweep/`. `--summary-only` rebuilds `comparison.csv` from available per-configuration summaries, but the script still validates HMOG configuration and the validation-user file.

## Touchalytics limitations

The default 180-row and raw-event 204-row results are separate experiments. The current collected default driver has loader/window/output-name differences from the historical default log, so CSV reanalysis is not proof of its exact producing source. The full sequential sweep log is preserved, but its selected per-user output is absent; the new reporter does not reconstruct that p-value or fill that gap with another experiment's CSV.

The new reanalysis reports each model/category separately. Unadjusted exploratory GRU results need not share the VanillaRNN nonsignificance. The diagnostic HMOG/Touchalytics comparison CSV still has blank HMOG cells and is not a completed two-dataset comparison.

## Execution records

For any future run, retain the source identity, command, working directory, actual interpreter/dependency versions, input hashes, exit status, stdout/stderr, timing and output comparisons. Keep failures and skips distinct from passes. Existing archived records are unchanged; new local execution reports need not be committed to the code repository.

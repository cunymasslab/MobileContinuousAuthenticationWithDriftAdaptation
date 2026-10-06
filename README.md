# Window-Based Incremental Adaptation for Behavioral Drift in Continuous Mobile Authentication

Replication code for evaluating window-based incremental adaptation under behavioral drift in continuous smartphone authentication.

This repository accompanies the manuscript by Avinash Jairam, Hui Chen, and Theodore Raphan. It studies whether later-session and cross-condition behavior degrades frozen authentication models, whether incremental adaptation recovers performance, whether that benefit exceeds additional labeled-data exposure alone, and how RTRL and BPTT compare under the same finite-window update protocol.

## Experimental protocol

The primary experiments use HMOG smartphone motion-sensor data. For each activity-posture condition, the first three available sessions provide offline-training data and the condition-specific fourth session provides the streaming evaluation data. Streams are balanced between legitimate and impostor windows. Each model scores a non-overlapping 100-timestep window before any update is made.

M2, M4, M7, and M8 then receive the window's ground-truth label and perform one update after every window. M3 is the exception: it scores every window but updates only after legitimate-user windows. The primary RTRL and BPTT variants reset recurrent state at window boundaries; M9 separately explores persistent state.

The drift categories are temporal, cross-posture, cross-activity, and combined activity-posture shift (`cross-both` in the code). Cross-condition comparisons also progress to a later session, so they are compound shifts rather than pure causal isolations of posture or activity.

## Model roles

| Model | Architecture | Streaming update | Role |
|---|---|---|---|
| M1 | VanillaRNN | None | Frozen control |
| M2 | VanillaRNN | RTRL on all labeled windows | Architecture-gradient comparison |
| M3 | VanillaRNN | RTRL on legitimate windows only | Adaptation-data ablation |
| M4 | VanillaRNN | BPTT on all labeled windows | Architecture-gradient comparison |
| M5 | LSTM | None | Frozen LSTM comparison |
| M6 | GRU | None | Frozen control |
| M7 | GRU | RTRL on all labeled windows | Architecture-gradient comparison |
| M8 | GRU | BPTT on all labeled windows | Architecture-gradient comparison |
| M9 | VanillaRNN | Persistent-state RTRL | Separate exploratory analysis |

VanillaRNN and GRU provide two recurrent architecture levels under common protocols; they are not an exhaustive search for the best sequence model. RTRL and BPTT are alternative gradient-computation mechanisms, not different update schedules.

## Assumptions and limitations

- Streaming adaptation is oracle supervised: every evaluated update uses a ground-truth label.
- Streams are balanced by construction rather than by deployment prevalence.
- Stream impostors are known identities with new observations, not previously unseen attackers.
- Validation and final-evaluation users are disjoint as target users, but not across every experimental role.
- The 50-combination drift-verification analysis is a selected subset and excludes cross-posture pairs.
- M9 and the Touchalytics analyses are exploratory; the selected Touchalytics sweep result is post-selection.
- The study does not report measured smartphone latency, memory use, energy consumption, or deployment readiness.

## Package status and reproduction

The repository now includes the accepted repaired factorial summary, its saved M7 repair/acceptance reports and statistical outputs, matched-stream results, M9 sweep summaries, drift-verification results, and separate Touchalytics diagnostic/adaptation results and selection logs. These are preserved outputs, not raw source datasets.

| Reference package | Contents and present verification scope |
|---|---|
| [Repaired factorial and M7–M8](artifacts/reference/m7_repaired_20260901/PACKAGING_NOTES.md) | Accepted 9,558-row summary; recorded M7–M8 reanalysis retains 9,216 configurations across 89 target users. Four statistical CSVs match exactly; the report matches after input-path normalization. |
| [Matched-stream and M9](artifacts/reference/matched_stream_m9/PACKAGING_NOTES.md) | Matched-stream summary/failures and saved report; five M9 sweep summaries and comparison. The report matches except for coverage-line ordering; the M9 comparison CSV matches exactly. |
| [Drift and Touchalytics diagnostics](artifacts/reference/drift_touchalytics_diagnostics/PACKAGING_NOTES.md) | Two 50-combination HMOG summaries and two Touchalytics diagnostic files. Identities match retained inventories; this import did not rerun statistical analyses or figures. |
| [Touchalytics adaptation](artifacts/reference/touchalytics_adaptation/PACKAGING_NOTES.md) | Separate default and raw-event results, full sweep/run logs, and source snapshots. Historical identities reconciled; separate CSV reanalysis is now available. The default producing-version mismatch remains unresolved. |

The [3–4 October execution records](docs/evidence/reproduction_20261004/README.md) document the earlier three completed reanalysis checks and their limitations in the recorded environment. Matched-stream mixed-effects warnings remain unresolved, and its report comparison is not byte-identical. M9 sweep reproduction and the separate user-level inference are distinct routes. File-identity checks do not establish full experimental reproduction.

Start with [the reproduction guide](docs/REPRODUCING.md) for the current commands and their limits. A fresh source archive matching commit `a017f58ef14a31d50b30c5b9e37fd8c8a1281548` was tested on 2026-10-05: four numerical-test programs, six artifact-analysis commands, and five direct publication-rendering commands completed. This was a source-snapshot test in an existing environment, **not** a fresh dependency installation, full Git-history review, or raw-data retraining.

The newer analysis entry points are:

| Command from the repository root | Scope |
|---|---|
| `python -I -B analysis/reproduce_summary_tables.py` | Two descriptive tables; standard-library only; differences calculated before rounding. |
| `python -I -B analysis/reproduce_m9_user_statistics.py` | Matched M9/M2 user-level paired t-test and t-based interval; exploratory/post-selection. |
| `python -I -B analysis/reproduce_touchalytics_statistics.py` | Separate default/raw-event summaries and Wilcoxon comparisons; sweep log observations, not regenerated selected-sweep inference. |

These three programs print to stdout by default, validate fixed input hashes, and require a new directory for optional `--output-dir` output. They do not need a paper PDF or manuscript source. The [publication renderers](analysis/publication_figures/README.md) remain reconstructed presentation variants, not exact historical image reproduction.

**Environment-sensitive identity:** the historical M7–M8 four-CSV exact match remains established in its recorded environment. In the 2026-10-05 environment, the two user-value CSVs matched exactly; the two statistical CSVs differed in 12 last-digit cells, with maximum absolute difference `3.47e-18`. That strict four-file comparison did not pass in the newer environment. The report still matched after input-path normalization; the M9 sweep CSV matched exactly, and the matched-stream report retained only its documented coverage-order exception. No reference or tolerance was changed to force a pass.

Reproduction is divided into:

1. numerical, unit, and packaging checks;
2. reanalysis of preserved canonical outputs; and
3. full training and streaming evaluation from source datasets.

Passing one level does not establish the others. See [the reproduction guide](docs/REPRODUCING.md), [the paper-to-code map](docs/PAPER_CODE_MAP.md), and [the provenance ledger](PROVENANCE.md).

## Data and release metadata

Raw HMOG sensor data, raw Touchalytics events, and feature matrices are not distributed here. The repository does include derived experiment results with dataset participant codes. Obtain source data from the authoritative providers and review the applicable terms; the package does not itself grant data-redistribution permission. Configure HMOG with an untracked local setting such as:

```dotenv
HMOG_ROOT=/path/to/your/hmog/data
```

Historical model IDs, algorithm names, output columns, and artifact filenames are retained for compatibility. No DOI, acceptance status, or release date is asserted until verified metadata is supplied.

The current test-entry-point fixes make GRU failures return a nonzero process status and make the persistent-gradient test extract its single loss value explicitly. They do not change core models, numerical tolerances, reference results, or the historical evidence directories.

## License

This project's original software and associated usage documentation are licensed under the [MIT License](LICENSE). The notice uses "The contributors" rather than individual names. This wording does not assign ownership to a new organization or replace any existing third-party notice.

Third-party code and dependencies retain their own applicable licenses and notices. The MIT license for the software does not license the HMOG or Touchalytics datasets, derived result files, historical experiment logs, or manuscript/figure material. It does not grant additional data-redistribution permissions. Data and artifact permissions must be considered separately before public distribution.

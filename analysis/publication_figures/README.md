# Tested reconstructed publication renderers

These five scripts are new presentation reconstructions prepared on 2026-10-04,
not recovered historical publication source. They are byte-for-byte copies of the
drafts executed on Compute B and reviewed with their original counterparts.
The original top-level repository plotting programs remain unchanged.

The [evidence and scope](../../docs/evidence/publication_rendering_20261004/README.md)
record exact equality of the captured numerical plotting quantities for all five
original/revised pairs. The seven manuscript PNGs are **not pixel-identical** to
these outputs. Visual review found consistent labels and visible data patterns,
with documented differences in spacing, dimensions and gray tones. This is not
permission to replace manuscript images or claim exact historical rendering.

## Inputs and output behavior

Paths in the input column are relative to the repository root. Output paths are
relative to the working directory, not this source directory.

| Program | Preserved input | Outputs |
|---|---|---|
| `generate_drift_detection_figure.py` | `artifacts/reference/drift_touchalytics_diagnostics/results_stage4b_exp1/results_drift_detection.csv` | `figures/drift_detection_boxplots.png` and `.pdf` |
| `generate_drift_gap_figure.py` | `artifacts/reference/drift_touchalytics_diagnostics/results_stage4b_exp1/results_stage4b_cv_summary.csv` | `figures/drift_gap_distribution.png` and `.pdf` |
| `generate_forgetting_histogram.py` | `artifacts/reference/m7_repaired_20260901/results_stage4_merged_summary_m7_repaired_20260901.csv` | `figures/forgetting_histogram_m2.png` and `.pdf` |
| `generate_m9_sweep_figure.py` | `artifacts/reference/matched_stream_m9/results_m9_lr_sweep/comparison.csv` | `figures/m9_lr_sweep.png` and `.pdf` |
| `generate_confound_figures.py` | Copy `artifacts/reference/matched_stream_m9/results_confound/results_confound_summary.csv` to `results_confound/results_confound_summary.csv` in the new working directory | Six PNG/PDF pairs in `results_confound/figures/`; only `paired_bar_v2`, `benefit_diff_vs_mmd_v2` and `benefit_diff_vs_ks_v2` are manuscript image files |

These scripts overwrite their fixed output names. Do not run them from an
existing result or reference directory. The original embedded usage comments
refer to the historical layout; the isolated staging below takes precedence.

## Environment and isolated invocation template

The recorded rendering runs used CPython 3.13.2, NumPy 2.2.6, Pandas 2.2.3,
SciPy 1.15.3, Matplotlib 3.10.3 and Pillow 12.3.0. See the existing
[reanalysis environment record](../../docs/evidence/reproduction_20261004/README.md).
This is not the historical model-training environment. Do not upgrade packages
to pursue a pixel match. No plotting run is required merely to import this folder.

The script bodies were originally tested through the capture helper. The five
relocated direct invocations also completed in the October 5 source-snapshot
workflow (Python 3.13.5, NumPy 2.3.5, Pandas 2.2.3, SciPy 1.17.0 and
Matplotlib 3.10.8). That was an existing environment, not a fresh installation.
It does not add a claim of pixel-identical manuscript reproduction. Run from the root of a disposable
Linux checkout with the recorded reanalysis environment activated; verify the
reference manifests as described in [the central guide](../../docs/REPRODUCING.md).

```bash
set -euo pipefail
umask 077
REPO="$(pwd -P)"
PYTHON="$(python -I -c 'import sys; print(sys.executable)')"
WORK="$(mktemp -d)"
mkdir "$WORK/results_confound"
cp "$REPO/artifacts/reference/matched_stream_m9/results_confound/results_confound_summary.csv" \
   "$WORK/results_confound/results_confound_summary.csv"
chmod 400 "$WORK/results_confound/results_confound_summary.csv"
export MPLCONFIGDIR="$WORK/matplotlib_config"
cd "$WORK"
"$PYTHON" -I -B "$REPO/analysis/publication_figures/generate_drift_detection_figure.py" \
    "$REPO/artifacts/reference/drift_touchalytics_diagnostics/results_stage4b_exp1/results_drift_detection.csv"
"$PYTHON" -I -B "$REPO/analysis/publication_figures/generate_drift_gap_figure.py" \
    "$REPO/artifacts/reference/drift_touchalytics_diagnostics/results_stage4b_exp1/results_stage4b_cv_summary.csv"
"$PYTHON" -I -B "$REPO/analysis/publication_figures/generate_forgetting_histogram.py" \
    "$REPO/artifacts/reference/m7_repaired_20260901/results_stage4_merged_summary_m7_repaired_20260901.csv"
"$PYTHON" -I -B "$REPO/analysis/publication_figures/generate_m9_sweep_figure.py" \
    "$REPO/artifacts/reference/matched_stream_m9/results_m9_lr_sweep/comparison.csv"
"$PYTHON" -I -B "$REPO/analysis/publication_figures/generate_confound_figures.py"
printf 'New outputs: %s\n' "$WORK"
```

The matched-stream script performs its existing 10,000-resample bootstrap and
produces three additional supporting plots. Successful PNG review does not
establish comparison to reference PDFs. Other paper-result routes, fresh-install validation,
licensing, privacy review and final submission readiness remain separate.

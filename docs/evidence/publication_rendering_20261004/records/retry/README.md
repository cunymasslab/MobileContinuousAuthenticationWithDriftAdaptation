# Capture-helper correction: targeted retry

Only drift_detection and matched_stream are executed in this retry.
The prior three exact numerical-trace matches are not rerun.
Unused signature binding was removed from the capture helper.
Arguments still go unchanged to Matplotlib; comparison criteria are unchanged.
All original/draft plotting scripts, inputs and reference images are unchanged.
Earlier output folders and failure records remain untouched.
Original bundle SHA-256: e2fb4083534f8b0c4baa6bc5bd337ec6a53ef9b71db6d4e31cfdb7ae94b3e344
Original helper SHA-256: 793c18afe5532b6af5b73e924e6883aa0bbf223076d286f8793668e134dc2a04
Corrected helper SHA-256: 34e15d83021aea60ca9a3a9c12542145f757a606653ded3bac493d10301373a3
Prior output ZIP SHA-256: 7980ae4f06eb960768313adb479fc5e906e7caa14562d6255e0ed46754463f96

Original README retained below; its full-batch scope is superseded here.

---

# Publication-rendering reconstruction — private test bundle

Prepared 4 October 2026. This is a test copy, not a repository upload and not
permission to replace manuscript figures. The five scripts under
`publication_draft/` are newly reconstructed presentation revisions; they are
not recovered historical publication-rendering source.

## What changed

- Grayscale bars/boxes and black reference lines, with distinct scatter marker
  shapes and Control/Drift hatching for the matched-stream plots.
- Publication-reference title, legend, and axis wording where documented in
  the supplied image review; no data categories were renamed in the CSVs.
- Overall titles moved inside the canvas and space reserved above multi-panel
  axes, addressing the clipped drift-detection title.
- PNG export set to 300 DPI, with tight saved bounding boxes; PDF exports retained.
- The obsolete hard-coded `Results-text claim: ~9.4%` console line is removed.
  The forgetting-rate computation is unchanged. No replacement result is hard-coded.

`PRESENTATION_CHANGES.diff` shows all edits; `PRESENTATION_EDITS.json` records the
exact ordered substitutions. The original source snapshots are unchanged.
Data loading, row filtering, histogram bin counts, category order, thresholds,
bootstrap seed/resample count and resampling order, regression, and value lookup
were not edited. The full matched-stream main function and numerical helpers
have identical abstract syntax between versions; the static record lists the
verified functions. Other edits occur within mixed computation/rendering main
functions and are documented as exact substitutions, not a false claim that
those entire functions are byte-identical.

The manuscript reference PNGs and five existing CSV inputs are copied unchanged
from the previous checked bundle. Original source commit:
`5cda80ad8d53777e7c458b8bfba9589b1471c636`.
Manuscript `main.tex` SHA-256:
`9bf753636ae7674f99b705c0b7699fbf79aae14cb3502b990f6e0398e850e6ba`.
Original failed figure-run ZIP SHA-256:
`14c039c500646fe274d7944c3209ed15fe78b39b09d98649a01632b5b49664ff`.
The previous failures remain failures of that exact image check, not overwritten
or retrospectively marked successful by this test.

## Tests prepared, not yet completed

The Python files were syntax-parsed. The orchestrator has been run in
`--check-only` mode only. A local synthetic-data smoke test checked that the
instrumentation ignores color/marker/DPI changes while detecting a changed bar
height. No original or draft research renderer was executed during preparation.

Run the real check only in the previously prepared Compute B environment:
CPython 3.13.2, NumPy 2.2.6, Pandas 2.2.3, SciPy 1.15.3,
Matplotlib 3.10.3, Pillow 12.3.0. No package upgrade or installation is needed.

After verifying the ZIP and extracting it into a new directory:

```text
<reanalysis-python> -I -B -u run_publication_figure_check.py --work-parent <existing-parent>
```

The runner refuses mismatched bundle hashes or environment versions, and makes
new private directories for all runs. Each task executes the original and draft
script once with identical copied inputs. Each invocation is limited to 600
seconds. The plotting scripts' statistical summaries and bootstrap calculations
are executed for this regression check; no model training, raw-data processing,
or completed M7–M8 inference is repeated.

No file is written to the original repository or the manuscript. Staged sources
and inputs are read-only; their hashes are checked after each execution.
The matched-stream program's three supporting plots are retained as supporting
outputs, not counted among the seven manuscript image files.

## Two distinct comparisons

1. **Original versus draft numerical plot payload.** An identical instrumentation
   wrapper records histogram counts/bin edges, boxplot coordinates, bar heights
   and error-bar widths, scatter-point offsets, fitted-line coordinates, and
   horizontal/vertical reference lines. These records are compared exactly.
   This checks preservation of the computations represented in the plots, in
   the tested environment. It does not establish statistical validity or exact
   agreement with unretained historical confidence-interval endpoints.
2. **Draft versus unchanged manuscript PNG.** File and decoded RGBA pixel equality
   are still recorded without a tolerance, resizing, cropping, or normalization.
   A difference continues to mean visual review required. Styling reconstruction
   and platform/font/rendering differences can prevent pixel equality. We do not
   assume that this new revision is the exact historical renderer.

The instrumentation does not consume random values or alter plot data. Its
code is included for review. Title-box measurements are written separately and
are only diagnostics; final title/clipping assessment requires viewing outputs.
PDFs are retained but are not compared with historical reference PDFs.

Exit 3: execution, input-integrity, or original/draft numerical comparison fails.
Exit 2: those numerical/execution checks pass but at least one strict manuscript
pixel comparison differs. Preserve outputs for visual review; do not force a pass.
Exit 0: those checks and all strict manuscript pixel comparisons pass; this still
does not establish full experimental or statistical-inference reproduction.

The output ZIP includes logs, both variants' images and numerical traces, unchanged
manuscript references, source snapshots, the comparison report, and an output
manifest. It excludes CSV input copies, caches, fonts and environment binaries.
Output logs contain local paths and need inspection before repository upload.

## Preservation

Do not modify the original input ZIP, the uploaded first-run output ZIP, current
repository plotting scripts, or manuscript figures while testing this draft.
Do not repeatedly change colors, dimensions, or comparison rules to make a test
appear successful. A documented visual/numerical reproduction outcome must remain
distinct from a claimed exact historical rendering.

This bundle does not resolve separate M9 user-level inference, table reproduction,
software licensing, a full repository/history privacy audit, or submission readiness.
`BUNDLE_MANIFEST.sha256` covers every other file, relative to this bundle directory.

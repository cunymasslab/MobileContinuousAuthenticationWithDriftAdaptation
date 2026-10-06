# Publication-rendering review — 4 October 2026

## Decision and limits

The five [reconstructed renderers](../../../analysis/publication_figures/README.md)
are suitable to preserve as tested **presentation variants**. Exact equality of
captured numerical plot data is established between original and draft programs
on the supplied inputs in the recorded environment. All seven unchanged manuscript
PNG references still fail strict pixel equality. That failure is preserved, not
overridden or relabeled as success. No exact historical source recovery is claimed.

The reviewer inspected seven revised/reference PNG pairs. Titles and axis/legend
wording are visible and consistent, including Control/Drift labels and the
thresholds. No visual discrepancy in the represented data patterns was identified.
Residual dimensions, spacing, gray tones, relative text sizes and rasterization
remain different. Visual inspection cannot recover exact historical bootstrap
endpoints or thousands of historical point coordinates from pixels.

The corrected drift-detection title is fully visible; its layout record is inside
the nominal canvas (top 0.98 instead of 1.02). The obsolete `~9.4%` console line is
absent from the tested forgetting draft, while the numerical trace is unchanged.
Existing manuscript figures are retained as references, not replaced.

## Archives inspected

| Upload | ZIP SHA-256 | Original output-manifest checks |
|---|---|---|
| `publication_first_run.zip` | `7980ae4f06eb960768313adb479fc5e906e7caa14562d6255e0ed46754463f96` | 72/72 |
| `publication_targeted_retry.zip` | `8ffa754d8764b463c155945abdb78fb64bb0c726bf74cd5e875e88b386c51a9d` | 57/57 |

The uploaded archives remain unchanged outside this package. `records/first/` and
`records/retry/` preserve selected evidence from each; the first failed attempt is
not discarded. [PACKAGING_INDEX.json](PACKAGING_INDEX.json) lists every included
source-to-package mapping and omitted archive entry. Original manifests are
historical identity records, not validators for this curated/sanitized layout.
Use [PACKAGED_MANIFEST.sha256](PACKAGED_MANIFEST.sha256) from the repository root.

## Numerical comparisons

Both variants of every successful pair returned 0 with no recorded changed staged
files or missing expected outputs. I independently compared the supplied trace
bytes, rather than relying solely on each batch's Boolean summary. No numerical
tolerance, reordering, rounding or normalization was used.

| Plotting task | Successful archive | Original/draft trace comparison | Trace SHA-256 |
|---|---|---|---|
| drift_detection | retry | Exact byte equality | `f415a0a468b063cd0b11b69d37383f9ab2958e694428be3cbf2a2a2af3b0f754` |
| drift_gap | first | Exact byte equality | `9a3f6e62b81385c1df5456d2b6fadf93706155c9b0b266e2026b9c35a23109eb` |
| forgetting | first | Exact byte equality | `dd046b70982b4e97539c6aca2835e571cdbd43a92f366f454a252bb943f853c2` |
| matched_stream | retry | Exact byte equality | `fcb57831c5bbcb1e99dba16a63577876ca85a0968b55c01fe99f86f4bb299ef5` |
| m9_sweep | first | Exact byte equality | `51c37f7622f8d9c76a3f0f0a59a747f5e47a02b8242790a7b9b121d2db946b8a` |

The helper captures histogram counts/bin edges; box/whisker, median and outlier
coordinates; bar rectangles and supplied error-bar widths; scatter-point offsets;
and fitted/reference-line coordinates. Matched-stream traces include six PNGs,
three of which are supporting outputs rather than manuscript images. These are
checks of captured quantities, not proof of every internal calculation or an
independent scientific validation of the study's assumptions.

The first run failed on boxplots because the original capture helper attempted
signature binding before Matplotlib could process `labels=`. The targeted retry
removed only that unused check. [CAPTURE_HELPER_FIX.diff](CAPTURE_HELPER_FIX.diff)
and both helper snapshots preserve the correction. All plotting source snapshots,
CSV identities and comparison criteria remained unchanged for the retry. The
synthetic compatibility-check success was reported in the supplied terminal
transcript; its separate toy files and launcher log were not in these two archives
and are not represented as included here.

## Strict manuscript-image comparisons

| Image | Reference pixels | Draft pixels | Exact decoded pixels equal? |
|---|---:|---:|---|
| `drift_detection_boxplots.png` | 1950 x 774 | 3870 x 1426 | No |
| `drift_gap_distribution.png` | 1200 x 750 | 2370 x 1466 | No |
| `forgetting_histogram_m2.png` | 2172 x 1395 | 2370 x 1466 | No |
| `paired_bar_v2.png` | 2172 x 1368 | 2355 x 1460 | No |
| `benefit_diff_vs_mmd_v2.png` | 2171 x 1484 | 2355 x 1604 | No |
| `benefit_diff_vs_ks_v2.png` | 2169 x 1484 | 2352 x 1604 | No |
| `m9_lr_sweep.png` | 2172 x 1395 | 2370 x 1466 | No |

All PNGs have approximately 300-DPI metadata. Identical DPI metadata does not mean
identical pixel dimensions. Five manuscript references identify Matplotlib 3.10.8;
the drafts identify 3.10.3. This metadata is not sufficient to attribute every
visible difference to a library version. No environment was altered to force a match.

Individual visual observations and image hashes are in [REVIEW.json](REVIEW.json).
Reference PNGs are under `manuscript_reference/`; unchanged reviewed draft PNGs are
under `reviewed_draft_png/`. The human-review display resizes copies for convenient
viewing only. Resizing was not part of the strict comparison.

## Preserved source and evidence

The original source commit reported by the bundle is
`5cda80ad8d53777e7c458b8bfba9589b1471c636`. Both output archives' renderer snapshots
match the previously supplied test bundle. Their reference PNGs also match that
bundle. This review did not recompile a manuscript PDF or re-establish a
source-to-PDF relationship.

All five original snapshots are retained in `source_snapshot/original/`.
The runnable reconstructed copies are separately located under
`analysis/publication_figures/`; none of the top-level plotting programs is replaced.
[PRESENTATION_CHANGES.diff](PRESENTATION_CHANGES.diff),
[PRESENTATION_EDITS.json](PRESENTATION_EDITS.json) and
[STATIC_CHECKS.json](STATIC_CHECKS.json) preserve the declared presentation changes.
The numerical source functions listed in the static record were rechecked by
syntax-tree comparison during this packaging review.

Both batch reports, maps, input manifests, all 14 execution logs, all 10 available
numerical traces and all 10 available title-layout records are retained. The
new report explains why successful evidence comes from two separate archives;
neither original batch summary was retrospectively rewritten as an all-task pass.

## Packaging and next validation

The only edits to copied text are replacement of the private server-home prefix
in logs and reports with `<COMPUTE_B_HOME>`. Dates, warnings, results, exit statuses,
negative comparison outcomes and suffix paths remain unchanged. See
[PACKAGING_NOTES.md](PACKAGING_NOTES.md) for counts and hashes.

This review performed file/manifest checks, source inspection, trace comparison,
PNG decoding and visual inspection. No research renderer, bootstrap, regression,
model training, Git write or manuscript edit was executed here. PNGs were copied,
not redrawn. Generated PDFs and supporting-plot PNGs remain in the untouched private
archives; they were not reference-PDF compared and are omitted from this curated
addition. This is not the entire paper-result, fresh-checkout, privacy or submission
readiness sign-off.

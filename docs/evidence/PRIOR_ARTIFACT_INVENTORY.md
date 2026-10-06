# Prior Artifact Inventory

This index preserves four excerpts supplied from retained terminal-output transcripts from earlier server provenance work. These are prior recorded observations, not scientific artifact bytes available in this checkout. Each hash below is a **SHA-256 recorded by the prior inventory**; it was not recomputed from the scientific artifact during packaging. The complete original transcripts are not included here, and none of these checks was rerun during packaging.

## M9 comparison

| Field | Prior recorded observation |
|---|---|
| Execution location | Compute B |
| Source transcript | `Pasted text(20260826-222340).txt` |
| Source lines | 393–404 |
| Artifact | `results_m9_lr_sweep/comparison.csv` |
| Size | 486 bytes |
| CSV rows | 5 |
| SHA-256 recorded by the prior inventory | `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8` |
| Availability in this checkout | Underlying artifact unavailable |
| Rerun during packaging | No |

Recorded columns:

```text
model, lr_online, n_rows, n_with_eer, mean_overall_eer,
std_overall_eer, min_overall_eer, max_overall_eer
```

Displayed entries in the excerpt:

| Model | `lr_online` | `n_rows` | `n_with_eer` | `mean_overall_eer` |
|---|---:|---:|---:|---:|
| M2 | 0.002 | 360 | 330 | 0.1508250176799398 |
| M9 | 0.0001 | 360 | 330 | 0.3207815323847635 |

**Scope:** historical artifact metadata, schema, and displayed values. This excerpt does not independently establish the M9 user-level confidence interval or paired t-test.

## Touchalytics default results

| Field | Prior recorded observation |
|---|---|
| Execution location | Compute A |
| Source transcript | `Pasted text(20260826-230140).txt` |
| Source lines | 155–162 |
| Artifact | `results_touchalytics/touchalytics_adaptation_results.csv` |
| Size | 27,591 bytes |
| CSV rows | 180 |
| Unique users | 40 |
| Seeds | 42, 123, 456 |
| SHA-256 recorded by the prior inventory | `d80f8c2aac5c5cc874a9e91b55c8cf3d8cc90c075fa8ad820aa30d0680bb3636` |
| Availability in this checkout | Underlying artifact unavailable |
| Rerun during packaging | No |

**Scope:** historical file identity, row count, and user/seed coverage. This establishes the historically inspected filename, but does not by itself resolve its relationship to later driver filenames.

## Touchalytics selection log

| Field | Prior recorded observation |
|---|---|
| Execution location | Compute A |
| Source transcript | `Pasted text(20260826-230140).txt` |
| Source lines | 324–328 |
| Artifact | `touchalytics_sweep.log` |
| Size | 368,919 bytes |
| SHA-256 recorded by the prior inventory | `d40753430b2f576a7da51847ed7cb53b675775a1de581f67122c514b98bba3c5` |
| Availability in this checkout | Underlying artifact unavailable |
| Rerun during packaging | No |

**Scope:** historical log identity and size. The log's contents and selection conclusions are not reproduced by this metadata excerpt.

## Matched-stream summary

| Field | Prior recorded observation |
|---|---|
| Execution location | Compute B |
| Source transcript | `Pasted text(20260829-192749).txt` |
| Source lines | 380–386 |
| Artifact | `results_confound/results_confound_summary.csv` |
| Size | 4,019,780 bytes |
| Line count | 9,397 |
| Header present in transcript | Yes |
| SHA-256 recorded by the prior inventory | `f1a5f9843df2d2a59fafcddc191eab41a8da12a0f551070c49422e6548047bb1` |
| Availability in this checkout | Underlying artifact unavailable |
| Rerun during packaging | No |

**Scope:** historical file identity, size, line count, and header. The line count is retained as recorded and is not relabeled as a newly parsed data-row count. This record does not establish the paired-test or mixed-effects results.

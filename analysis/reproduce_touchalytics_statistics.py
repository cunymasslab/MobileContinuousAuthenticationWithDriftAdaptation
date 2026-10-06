"""Reanalyse saved default/raw Touchalytics results; display the retained sweep log.

Usage (Python 3.10+, NumPy, SciPy)::

    python -I -B analysis/reproduce_touchalytics_statistics.py
    python -I -B analysis/reproduce_touchalytics_statistics.py --output-dir /path/to/new-dir

Inputs default to artifacts/reference/touchalytics_adaptation relative to this
script's repository. --reference-dir accepts another copy of that same folder.
The two CSVs and sweep log must match their recorded SHA-256 identities. No data
are downloaded; no training driver or repository module is imported or executed.

DEFAULT AND RAW-EVENT REANALYSIS: the two experiments are kept separate. Average
M1-M8 EERs over the three seeds within (user_id, drift_type), then weight users
equally within each drift type. The four frozen/adapting comparisons, NumPy mean
operations, two-sided Wilcoxon test (zero_method='wilcox', method='auto', no
continuity correction), and near-zero skip rule follow the retained
source_snapshot/analyze_touchalytics_raw_results.py. Its SHA-256 is recorded in
METHOD_SOURCE_SHA256 below. Unlike its permissive loader, this new script rejects
invalid/missing values, duplicates, and incomplete seed groups; it never silently
drops rows. Input row order is preserved during aggregation. No data are rounded
before testing. Near-zero skipped comparisons are NOT labelled equivalent.

This is newly reconstructed reproduction code. Reusing the retained analyzer's
method on the historical default CSV does not recover that experiment's original
producing driver. See the reference folder's PACKAGING_NOTES.md for its distinct
loader/settings/output-name limitation and the preserved runtime warnings.

SWEEP DISPLAY ONLY: all four sequential round tables, winner/configuration lines,
and the final configuration are extracted from the hash-checked log as text.
Their means/p-values are log observations, NOT recomputed statistics. No selected
per-user sweep CSV is available in this reference set. The same data informed
selection and assessment; the nominal post-selection p-value is not independent
confirmatory evidence. No new selection or multiple-comparison correction occurs.

By default this script writes only stdout. Optional --output-dir creates a NEW
directory (existing parent required) for three CSVs and the sweep excerpt. It
refuses existing destinations and protected code/reference paths. No manuscript,
PDF, expected EER/p-value, or precomputed result table is embedded or required.
This does not recalculate EERs from raw scores, repeat drift detection, create a
completed HMOG/Touchalytics comparison, train models, plot, or test equivalence.

Wilcoxon ties/zeros and automatic p-value methods depend on the SciPy version;
the actual interpreter and versions are printed. The separately recorded
reanalysis environment uses NumPy 2.2.6 and SciPy 1.15.3. This is not a claim that
those were the original experiment versions. API documentation:
https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.wilcoxon.html
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPOSITORY_ROOT / "artifacts/reference/touchalytics_adaptation"
INPUTS = {
    "default": ("results_touchalytics/touchalytics_adaptation_results.csv",
                "d80f8c2aac5c5cc874a9e91b55c8cf3d8cc90c075fa8ad820aa30d0680bb3636"),
    "raw_event": ("results_touchalytics_raw/touchalytics_raw_adaptation_results.csv",
                  "7f3da80dd06c59795fbd4155f0fe642111142f7d1898d65ecb775991ab3ae212"),
    "sweep_log": ("touchalytics_sweep.log",
                  "d40753430b2f576a7da51847ed7cb53b675775a1de581f67122c514b98bba3c5"),
}
# Identity of the inspected methodological source, not an executable dependency.
METHOD_SOURCE_SHA256 = "6b5261924a41e14322130243d54e48f1913dd63988db1fbef4b0fec40538ea31"
MODELS = tuple(f"M{i}" for i in range(1, 9))
DRIFTS = ("temporal", "cross_activity", "cross_both")
SEEDS = {"42", "123", "456"}
PAIRS = (("M1", "M2"), ("M1", "M4"), ("M6", "M7"), ("M6", "M8"))
KEY = ("user_id", "drift_type", "seed")
BASE_COLUMNS = (*KEY, "n_train_windows", "n_test_windows",
                *(f"{m}_overall_eer" for m in MODELS))


def load_inputs(reference: Path) -> dict[str, bytes]:
    contents = {}
    for name, (relative, expected) in INPUTS.items():
        data = (reference / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SHA-256 differs for {name}; no reanalysis performed")
        contents[name] = data
    return contents


def parse_csv(data: bytes, family: str) -> dict[tuple[str, str], list[dict]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
    expected = BASE_COLUMNS + (("adaptation_benefit",) if family == "default" else ())
    if reader.fieldnames != list(expected):
        raise ValueError(f"Unexpected, missing or duplicate {family} columns")
    groups: dict[tuple[str, str], list[dict]] = {}
    seen = set()
    for row in reader:
        if None in row or any(v is None or v == "" for v in row.values()):
            raise ValueError(f"Malformed or incomplete {family} row near {reader.line_num}")
        if any(row[k] != row[k].strip() for k in KEY):
            raise ValueError("Padded key; no automatic correction")
        if row["drift_type"] not in DRIFTS or row["seed"] not in SEEDS:
            raise ValueError("Unknown drift type or seed")
        if not row["user_id"].isdigit():
            raise ValueError("Unexpected user identifier")
        key = tuple(row[k] for k in KEY)
        if key in seen:
            raise ValueError("Duplicate key; no automatic deduplication")
        seen.add(key)
        for name in ("n_train_windows", "n_test_windows"):
            if not row[name].isdigit() or int(row[name]) <= 0:
                raise ValueError("Invalid window count")
        for model in MODELS:
            col = f"{model}_overall_eer"
            value = float(row[col])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {family} EER")
            row[col] = value
        if family == "default":
            value = float(row["adaptation_benefit"])
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("Invalid saved adaptation-benefit field")
            # Recompute every comparison from model EERs, not this stored field.
        groups.setdefault((key[0], key[1]), []).append(row)
    if not groups or {drift for _, drift in groups} != set(DRIFTS):
        raise ValueError("Empty result or missing drift category")
    for rows in groups.values():
        if len(rows) != len(SEEDS) or {r["seed"] for r in rows} != SEEDS:
            raise ValueError("Incomplete seed group; no implicit reweighting")
    return groups


def analyse(groups: dict, family: str) -> tuple[list[dict], list[dict], list[dict]]:
    import numpy as np
    from scipy import stats

    users = []
    for (uid, drift), rows in groups.items():
        means = {f"{m}_mean_eer": float(np.mean([r[f"{m}_overall_eer"] for r in rows]))
                 for m in MODELS}
        users.append({"family": family, "user_id": uid, "drift_type": drift,
                      "n_seeds": len(rows), **means})
    model_means, comparisons = [], []
    for drift in DRIFTS:
        eligible = [r for r in users if r["drift_type"] == drift]
        n = len(eligible)
        model_means.append({"family": family, "drift_type": drift, "n_users": n,
                            **{f"{m}_mean_eer": float(np.mean([r[f"{m}_mean_eer"]
                                                              for r in eligible]))
                               for m in MODELS}})
        for frozen, adapting in PAIRS:
            a = np.asarray([r[f"{frozen}_mean_eer"] for r in eligible])
            b = np.asarray([r[f"{adapting}_mean_eer"] for r in eligible])
            delta = a - b
            # Preserve the retained analyzer's numerical skip rule, not its 'eq' label.
            skipped = bool(np.allclose(delta, 0, rtol=1e-5, atol=1e-8))
            statistic, pvalue = None, None
            if not skipped:
                test = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                                     alternative="two-sided", method="auto")
                statistic, pvalue = float(test.statistic), float(test.pvalue)
                if not math.isfinite(statistic) or not math.isfinite(pvalue):
                    raise ValueError("Nonfinite Wilcoxon output")
            comparisons.append({
                "family": family, "drift_type": drift, "frozen": frozen,
                "adapting": adapting, "n_users": n,
                "mean_benefit": float(np.mean(delta)), "median_benefit": float(np.median(delta)),
                "wilcoxon_statistic": statistic, "two_sided_pvalue_unadjusted": pvalue,
                "zero_method": "wilcox", "method_request": "auto", "correction": False,
                "test_status": "skipped_near_zero_not_equivalence" if skipped else "computed",
                "n_zero_differences": int(np.sum(delta == 0)),
                "users_adapting_lower": int(np.sum(delta > 0)),
                "users_frozen_lower": int(np.sum(delta < 0)),
                "mean_absolute_delta": float(np.mean(np.abs(delta))),
            })
    return users, model_means, comparisons


def sweep_excerpt(data: bytes) -> str:
    """Extract stored tables verbatim; do not infer results or rerun selection."""
    lines = data.decode("utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("ROUND:")]
    expected = ("window_size", "learning_rate", "feature_set", "hidden_dim")
    if len(starts) != len(expected):
        raise ValueError("Unexpected number of saved sweep rounds")
    excerpts = []
    for start, name in zip(starts, expected):
        if not lines[start].startswith(f"ROUND: {name}  (sweeping "):
            raise ValueError("Unexpected round order or label")
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith(">>> Current config:")), None)
        if end is None or end - start > 30:
            raise ValueError("Incomplete sweep table/configuration block")
        block = lines[start:end + 1]
        if not any(line.startswith("| ") for line in block):
            raise ValueError("Missing sweep candidate table")
        excerpts.append(f"[Saved sweep log lines {start + 1}-{end + 1}; NOT recomputed]\n"
                        + "\n".join(block))
    finals = [i for i, line in enumerate(lines) if line.startswith("FINAL BEST CONFIG")]
    if len(finals) != 1 or finals[0] <= starts[-1]:
        raise ValueError("Missing or repeated final sweep configuration")
    start = finals[0]
    final = lines[start:]
    keys = [re.match(r"^  (W|lr_online|feature_set|H):", line) for line in final]
    if [m.group(1) for m in keys if m] != ["W", "lr_online", "feature_set", "H"]:
        raise ValueError("Incomplete final configuration")
    excerpts.append(f"[Saved sweep log lines {start + 1}-{len(lines)}; NOT recomputed]\n"
                    + "\n".join(final))
    return "\n\n".join(excerpts) + "\n"


def output_location(path: Path, reference: Path) -> Path:
    expanded = path.expanduser()
    destination = expanded.resolve()
    if expanded.is_symlink() or destination.exists():
        raise ValueError("Output already exists; no overwrite permitted")
    if not destination.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    protected = [reference.resolve(), *(REPOSITORY_ROOT / p for p in ("artifacts", "analysis", "docs"))]
    if any(destination == p.resolve() or p.resolve() in destination.parents for p in protected):
        raise ValueError("Output must be outside protected reference/code directories")
    return destination


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Optional NEW directory for local outputs")
    args = parser.parse_args()
    try:
        reference = args.reference_dir.expanduser().resolve()
        destination = output_location(args.output_dir, reference) if args.output_dir else None
        contents = load_inputs(reference)
        groups = {family: parse_csv(contents[family], family) for family in ("default", "raw_event")}
        excerpt = sweep_excerpt(contents["sweep_log"])
        import numpy as np
        import scipy
        records = {family: analyse(grouped, family) for family, grouped in groups.items()}
        if destination is not None:
            destination.mkdir(mode=0o700, parents=False, exist_ok=False)
            for index, filename in enumerate(("touchalytics_user_means.csv",
                                              "touchalytics_model_means.csv",
                                              "touchalytics_paired_statistics.csv")):
                save_csv(destination / filename, [row for value in records.values() for row in value[index]])
            with (destination / "sweep_log_excerpt.txt").open("x", encoding="utf-8") as stream:
                stream.write("LOG OBSERVATIONS ONLY; NOT NEW INFERENCE. EXPLORATORY/POST-SELECTION.\n\n" + excerpt)
        print("TOUCHALYTICS: SEPARATE DEFAULT / RAW-EVENT REANALYSIS")
        print(f"Python {sys.version.split()[0]}; NumPy {np.__version__}; SciPy {scipy.__version__}")
        print("Method: seeds averaged within user/drift, then equal users within drift.")
        print("Two-sided Wilcoxon; zero_method=wilcox, correction=False, method=auto.")
        print("Unadjusted exploratory tests; no formal equivalence test or new selection.")
        for name, (relative, digest) in INPUTS.items():
            print(f"Input {name}: SHA-256 MATCH {digest}  {relative}")
        for family, (users, model_means, comparisons) in records.items():
            print(f"\n=== {family}: {sum(len(v) for v in groups[family].values())} CSV rows; "
                  f"{len({r['user_id'] for r in users})} distinct users; {len(users)} user/drift groups ===")
            print("Category         n " + " ".join(f"{m:>9}" for m in MODELS))
            for row in model_means:
                print(f"{row['drift_type']:<16} {row['n_users']:>2} "
                      + " ".join(f"{row[f'{m}_mean_eer']:9.6f}" for m in MODELS))
            print("Category         Pair     n   Mean benefit   Median benefit       W       p (unadjusted)")
            for row in comparisons:
                w, p = row["wilcoxon_statistic"], row["two_sided_pvalue_unadjusted"]
                test_text = f"{w:8.3f} {p:18.10g}" if p is not None else row["test_status"]
                pair = row["frozen"] + "->" + row["adapting"]
                print(f"{row['drift_type']:<16} {pair:<7} {row['n_users']:>3} "
                      f"{row['mean_benefit']:+14.10f} {row['median_benefit']:+16.10f} {test_text}")
        print("\n=== SEQUENTIAL SWEEP: STORED LOG OBSERVATIONS, NOT RECOMPUTED ===")
        print(excerpt)
        print("No selected-sweep per-user data are supplied; its p-value was NOT reproduced.")
        print("Default and raw-event CSVs are not substitutes for the selected-sweep data.")
        print("Producing-version mismatch and historical runtime warnings remain unresolved.")
        print(f"Local output directory: {destination}" if destination else "No files written (stdout only).")
        return 0
    except (OSError, ValueError, csv.Error, ImportError, UnicodeError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

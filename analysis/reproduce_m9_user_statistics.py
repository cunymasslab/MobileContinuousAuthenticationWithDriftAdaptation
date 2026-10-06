"""Reproduce the exploratory user-level M9-versus-M2 comparison from saved CSVs.

Usage (Python 3.10+ and SciPy 1.10+; no training/data-loader dependencies)::

    python -I -B analysis/reproduce_m9_user_statistics.py
    python -I -B analysis/reproduce_m9_user_statistics.py --output-dir /path/to/new-dir

Inputs default to artifacts/reference/matched_stream_m9/ relative to this
script's repository. --reference-dir accepts another copy of that same reference
folder; the two summary CSVs and source_snapshot/validation_users.txt must match
the fixed SHA-256 identities. No downloads or package installations are made.

Method: compare M9 and M2 at the previously selected online rate 0.002. Match
(target_user, train_condition, stream_condition, seed) keys exactly; require
identical missing-EER masks and successful-row stream metadata. Average usable
condition-pair/seed results within each validation user, then weight users equally.
SciPy's paired two-sided t-test and its 95% t-based confidence interval operate
on those user means, NOT on the individual configuration rows.

This is newly reconstructed reproduction code, not recovered historical analysis
source. The t-based interval is explicitly specified here; a matching rounded
interval alone does not identify the historical implementation. This script does
not perform a bootstrap, new learning-rate selection, multiple-comparison
correction, an equivalence test, training, plotting, or M7-M8 reanalysis.

The comparison remains exploratory/post-selection: these same validation users
were used for the learning-rate sweep. The unadjusted p-value and interval do
not constitute an independent confirmatory test. Matching summary metadata does
not verify per-window labels or ordering, which are not read by this script.

No manuscript, PDF, expected EERs, expected p-values or expected confidence limits
are embedded or read. By default only stdout is written. --output-dir saves two
CSV files in a NEW directory with an existing parent; existing directories and
paths within protected reference/artifact trees are refused. Inputs are read only.

API reference:
https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.ttest_rel.html
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import statistics
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPOSITORY_ROOT / "artifacts/reference/matched_stream_m9"
INPUTS = {
    "M2": ("results_m9_lr_sweep/M2_lr_0.002_summary.csv",
           "db0472d3736f5103ae812a842d8ea1dd5874546a95a89eb555c1a84c95b69ca1"),
    "M9": ("results_m9_lr_sweep/M9_lr_0.002_summary.csv",
           "f16fe8cdeaa1e4b2d23f122953c8eadfe6fa5ed76193a42f2c70d0c9015bdae6"),
    "users": ("source_snapshot/validation_users.txt",
              "e36af2ac3535f29ba0dea48ef62b49f45a6b1300c36d357ba510a305300cdb0f"),
}
KEY_COLUMNS = ("target_user", "train_condition", "stream_condition", "seed")
METADATA = ("pair_category", "train_sessions", "stream_session",
            "stream_legit_count", "stream_impostor_count", "total_windows")
ACTIVITIES = ("READING", "WRITING", "MAP")
POSTURES = ("SITTING", "WALKING")
SEEDS = ("42", "123", "456")
Key = tuple[str, ...]


def load_inputs(reference_dir: Path) -> dict[str, bytes]:
    result = {}
    for label, (relative, expected) in INPUTS.items():
        data = (reference_dir / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Input SHA-256 differs for {label}; no analysis performed")
        result[label] = data
    return result


def parse_summary(data: bytes, model: str) -> dict[Key, dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
    headers = reader.fieldnames
    required = set(KEY_COLUMNS + METADATA) | {f"{model}_overall_eer"}
    if not headers or len(headers) != len(set(headers)) or not required <= set(headers):
        raise ValueError(f"Missing/duplicate header or required columns in {model}")
    records = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed {model} CSV record near line {reader.line_num}")
        if any(not row[c] or row[c] != row[c].strip() for c in KEY_COLUMNS):
            raise ValueError(f"Empty/padded {model} key; no automatic correction")
        key = tuple(row[c] for c in KEY_COLUMNS)
        if key in records:
            raise ValueError(f"Duplicate {model} key; no automatic deduplication")
        if row["pair_category"] != "cross-both":
            raise ValueError(f"Unexpected {model} category")
        records[key] = row
    if not records:
        raise ValueError(f"Empty {model} summary")
    return records


def eer(row: dict[str, str], model: str) -> float | None:
    value = row[f"{model}_overall_eer"]
    if value == "":
        # Do not reinterpret partially populated target-model results as failures.
        if any(v != "" for k, v in row.items() if k.startswith(model + "_")):
            raise ValueError(f"Partial {model} result with empty overall EER")
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Invalid {model} overall EER")
    return number


def aggregate(data: dict[str, bytes]) -> tuple[list[dict], dict]:
    user_lines = data["users"].decode("utf-8").splitlines()
    if (not user_lines or any(not u or u != u.strip() for u in user_lines)
            or len(set(user_lines)) != len(user_lines)):
        raise ValueError("Invalid or duplicate validation-user identifiers")
    users = sorted(user_lines)
    records = {m: parse_summary(data[m], m) for m in ("M2", "M9")}
    pairs = [(f"{a}+{p}", f"{b}+{q}") for a in ACTIVITIES for p in POSTURES
             for b in ACTIVITIES for q in POSTURES if a != b and p != q]
    planned = {(u, a, b, s) for u in users for a, b in pairs for s in SEEDS}
    for model in records:
        if set(records[model]) != planned:
            raise ValueError(f"{model} attempted keys do not match the validation design")
    grouped: dict[str, list[tuple[float, float]]] = {u: [] for u in users}
    excluded = 0
    for key in sorted(planned):
        r2, r9 = records["M2"][key], records["M9"][key]
        a, b = eer(r2, "M2"), eer(r9, "M9")
        if (a is None) != (b is None):
            raise ValueError("Unmatched missing EER; no implicit inner join")
        if a is None:
            excluded += 1
            continue
        if any(r2[field] != r9[field] for field in METADATA):
            raise ValueError("Matched-key stream metadata differs")
        if any(not r2[field].strip() for field in METADATA):
            raise ValueError("Successful key has empty stream metadata")
        legit, impostor, total = (int(r2[c]) for c in
                                 ("stream_legit_count", "stream_impostor_count", "total_windows"))
        if legit <= 0 or legit != impostor or total != legit + impostor:
            raise ValueError("Invalid or unbalanced stream window counts")
        grouped[key[0]].append((a, b))
    values = []
    for user in users:
        pairs_for_user = grouped[user]
        if not pairs_for_user:
            raise ValueError("A validation user has no matched successful configurations")
        m2 = statistics.fmean(a for a, _ in pairs_for_user)
        m9 = statistics.fmean(b for _, b in pairs_for_user)
        values.append({"target_user": user, "n_configurations": len(pairs_for_user),
                       "M2_mean_eer": m2, "M9_mean_eer": m9,
                       "M9_minus_M2": m9 - m2})
    coverage = {"attempted_keys_per_model": len(planned),
                "matched_successful_keys": sum(len(v) for v in grouped.values()),
                "jointly_missing_eer_keys": excluded, "n_users": len(users)}
    return values, coverage


def infer(values: list[dict]) -> dict:
    # Lazy import: --help and input validation do not import the research stack.
    import numpy as np
    import scipy
    from scipy import stats

    m2 = np.asarray([r["M2_mean_eer"] for r in values], dtype=float)
    m9 = np.asarray([r["M9_mean_eer"] for r in values], dtype=float)
    delta = m9 - m2
    if len(delta) < 2 or not np.all(np.isfinite(delta)) or np.std(delta, ddof=1) == 0:
        raise ValueError("Paired t-test requires nondegenerate, finite user differences")
    result = stats.ttest_rel(m9, m2, alternative="two-sided", nan_policy="raise")
    if not hasattr(result, "confidence_interval"):
        raise ValueError("SciPy 1.10+ is required; use the separately recorded reanalysis environment")
    interval = result.confidence_interval(confidence_level=0.95)
    if not all(math.isfinite(float(v)) for v in
               (result.statistic, result.pvalue, interval.low, interval.high)):
        raise ValueError("Nonfinite inferential output")
    return {"comparison": "M9_minus_M2", "lr_online": 0.002,
            "weighting": "equal_user_after_within_user_configuration_mean",
            "n_users": len(values), "M2_mean_eer": float(np.mean(m2)),
            "M9_mean_eer": float(np.mean(m9)), "mean_difference": float(np.mean(delta)),
            "difference_eer_percentage_points": float(np.mean(delta) * 100),
            "paired_t_statistic": float(result.statistic), "df": int(result.df),
            "two_sided_pvalue": float(result.pvalue), "ci_method": "paired_t",
            "confidence_level": 0.95, "ci_low": float(interval.low),
            "ci_high": float(interval.high), "users_M2_lower": int(np.sum(delta > 0)),
            "users_M9_lower": int(np.sum(delta < 0)), "users_tied": int(np.sum(delta == 0)),
            "selection_status": "exploratory_post_selection_unadjusted",
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__, "scipy_version": scipy.__version__}


def output_location(path: Path, reference: Path) -> Path:
    destination = path.expanduser().resolve()
    if path.expanduser().is_symlink() or destination.exists():
        raise ValueError("Output already exists; no overwrite is permitted")
    if not destination.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    protected = [reference.resolve(), (REPOSITORY_ROOT / "artifacts").resolve(),
                 (REPOSITORY_ROOT / "analysis").resolve(), (REPOSITORY_ROOT / "docs").resolve()]
    if any(destination == p or p in destination.parents for p in protected):
        raise ValueError("Output must be outside protected code/reference directories")
    return destination


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_ROOT,
                        help="Another copy of the same accepted reference folder")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional NEW directory for user-level values and statistics CSVs")
    args = parser.parse_args()
    try:
        reference = args.reference_dir.expanduser().resolve()
        destination = output_location(args.output_dir, reference) if args.output_dir else None
        data = load_inputs(reference)
        values, coverage = aggregate(data)
        result = infer(values)
        if destination is not None:
            destination.mkdir(mode=0o700, parents=False, exist_ok=False)
            save_csv(destination / "m9_m2_user_level_values.csv", values)
            save_csv(destination / "m9_m2_user_level_statistics.csv", [{**coverage, **result}])
        print("M9-M2 EXPLORATORY USER-LEVEL COMPARISON (online rate 0.002)")
        for name, (relative, digest) in INPUTS.items():
            print(f"Input {name}: SHA-256 MATCH {digest}  {relative}")
        for name, value in coverage.items():
            print(f"{name}: {value}")
        for name, value in result.items():
            print(f"{name}: {value}")
        print("\nEXPLORATORY / POST-SELECTION: the same validation users informed rate selection.")
        print("The paired t-test and t-based interval are unadjusted, not confirmatory evidence.")
        print("No training, bootstrap, plotting or historical-source recovery is claimed.")
        print(f"CSV outputs: {destination}" if destination else "No files written (stdout only).")
        return 0
    except (OSError, ValueError, csv.Error, ImportError, UnicodeError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

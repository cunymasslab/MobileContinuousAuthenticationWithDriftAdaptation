"""Generate two descriptive tables from the accepted reference CSVs.

Usage from the repository root (Python 3.10+; standard library only)::

    python -I -B analysis/reproduce_summary_tables.py
    python -I -B analysis/reproduce_summary_tables.py --output-dir /path/to/new-tables

The default inputs are resolved relative to this script's repository, not the
current working directory. --factorial-csv and --matched-csv may point to other
copies of the SAME accepted files; SHA-256 identities are always checked.

Without --output-dir, only the terminal is written. With it, two CSVs are saved
in a new directory whose parent must already exist. Each contains unrounded
50-digit-context decimal calculations and separately formatted three-decimal
columns. Existing directories are never overwritten, and outputs inside the
repository's artifacts/ tree or either input's parent are refused.

Calculations preserve the descriptive-table rounding check's aggregation:
* tab:percategory: equal configuration weights within each drift category;
  exclude only jointly empty M1/M2 overall-EER pairs and reject partial pairs.
  This table-specific rule does not establish completeness of every model metric.
* tab:confound: average the saved control/drift benefits within each user and
  category, then weight users equally. The Overall row weights the four category
  estimates equally. Identical user sets across categories are required.
* Subtract unrounded values before one half-even rounding to three decimals.

This is new reproduction code, not recovered historical table-generation code.
It does not read a manuscript or PDF, and contains no expected table-cell values.
No training, plotting, bootstrap, or hypothesis tests are performed. The input
CSV files are read only. No repository modules or third-party packages are used.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import io
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_RELATIVE = Path(
    "artifacts/reference/m7_repaired_20260901/"
    "results_stage4_merged_summary_m7_repaired_20260901.csv"
)
MATCHED_RELATIVE = Path(
    "artifacts/reference/matched_stream_m9/results_confound/"
    "results_confound_summary.csv"
)
EXPECTED_HASHES = {
    "factorial": "7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098",
    "matched_stream": "f1a5f9843df2d2a59fafcddc191eab41a8da12a0f551070c49422e6548047bb1",
}
CATEGORIES = ("temporal", "cross-posture", "cross-activity", "cross-both")
KEYS = ("target_user", "train_condition", "stream_condition", "seed")
TABLE_ORDER = {
    "tab:percategory": CATEGORIES,
    "tab:confound": ("temporal", "cross-activity", "cross-posture", "cross-both", "overall"),
}
LABELS = {
    "temporal": "Temporal", "cross-posture": "Cross-posture",
    "cross-activity": "Cross-activity", "cross-both": "Cross-both", "overall": "Overall",
}
HEADERS = {
    "tab:percategory": ("M1_mean_eer", "M2_mean_eer", "eer_reduction"),
    "tab:confound": ("control_benefit", "drift_benefit", "difference"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mean(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("Empty mean: a required category has no usable rows")
    return sum(values, Decimal(0)) / Decimal(len(values))


def shown(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        return format(value.quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN), ".3f")


def rows(data: bytes) -> list[dict[str, str]]:
    """Parse a CSV without silently accepting malformed rows or duplicate keys."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Missing or duplicate CSV header names")
    required = set(KEYS) | {"pair_category"}
    if not required.issubset(fields):
        raise ValueError("Missing key/category columns: " + ", ".join(sorted(required - set(fields))))
    result = []
    seen = set()
    for record in reader:
        if None in record or any(value is None for value in record.values()):
            raise ValueError(f"Malformed CSV record near physical line {reader.line_num}")
        if any(not record[key].strip() for key in required):
            raise ValueError(f"Empty key/category near physical line {reader.line_num}")
        if any(record[key] != record[key].strip() for key in required):
            raise ValueError(f"Padded key/category near physical line {reader.line_num}")
        key = tuple(record[k] for k in KEYS)
        if key in seen:
            raise ValueError("Duplicate experiment key; no automatic deduplication")
        seen.add(key)
        if record["pair_category"] not in CATEGORIES:
            raise ValueError("Unexpected drift category")
        result.append(record)
    if not result:
        raise ValueError("Input CSV has no data rows")
    return result


# The calculation below is retained from the prior descriptive rounding check.
# Input validation and the command-line/output interface are separate from it.
def _calculate(factorial: bytes, confound: bytes) -> tuple[dict, dict]:
    f, c = rows(factorial), rows(confound)
    grouped = defaultdict(list)
    skipped = 0
    for r in f:
        values = [r['M1_overall_eer'].strip(), r['M2_overall_eer'].strip()]
        if not any(values):
            skipped += 1
            continue
        if not all(values):
            raise ValueError('Partial M1/M2 pair; no automatic exclusion')
        a, b = map(Decimal, values)
        if not all(v.is_finite() and 0 <= v <= 1 for v in (a, b)):
            raise ValueError('Invalid nonempty factorial EER')
        grouped[r['pair_category']].append((a, b))
    main_table = {}
    for category in CATEGORIES:
        entries = grouped[category]
        a = mean([x[0] for x in entries])
        b = mean([x[1] for x in entries])
        main_table[category] = (a, b, a - b)
    by_user = defaultdict(list)
    for r in c:
        a, b = (Decimal(r[k]) for k in ('adaptation_benefit_nodrift', 'adaptation_benefit_drift'))
        if not all(v.is_finite() and -1 <= v <= 1 for v in (a, b)):
            raise ValueError('Invalid matched-stream benefit')
        by_user[(r['target_user'], r['pair_category'])].append((a, b))
    users_by_cat = {cat: {user for user, ct in by_user if ct == cat} for cat in CATEGORIES}
    if any(users != users_by_cat[CATEGORIES[0]] for users in users_by_cat.values()):
        raise ValueError('Category user sets differ; no silent weighting change')
    matched_table = {}
    for category in CATEGORIES:
        user_means = [(mean([a for a, b in v]), mean([b for a, b in v]))
                      for (user, cat), v in by_user.items() if cat == category]
        a = mean([v[0] for v in user_means]); b = mean([v[1] for v in user_means])
        matched_table[category] = (a, b, b - a)
    matched_table['overall'] = tuple(mean([v[i] for v in matched_table.values()]) for i in range(3))
    coverage = {
        'factorial_input_rows': len(f), 'factorial_M1_M2_both_empty_excluded': skipped,
        'factorial_paired_rows': sum(map(len, grouped.values())),
        'factorial_paired_users': len({r['target_user'] for r in f if r['M1_overall_eer'].strip()}),
        'factorial_category_rows': {k: len(v) for k, v in grouped.items()},
        'matched_stream_rows': len(c), 'matched_stream_users_per_category': {k: len(v) for k, v in users_by_cat.items()},
        'duplicate_keys_each_input': 0,
    }
    return {'tab:percategory': main_table, 'tab:confound': matched_table}, coverage


def calculate(factorial: bytes, confound: bytes) -> tuple[dict, dict]:
    """Apply the same precision independently of the caller's decimal context."""
    with localcontext() as context:
        context.prec = 50
        return _calculate(factorial, confound)


def output_text(tables: dict) -> str:
    lines = []
    for name, table in tables.items():
        lines += [name, "category | " + " | ".join(HEADERS[name])]
        for category in TABLE_ORDER[name]:
            lines.append(LABELS[category] + " | " + " | ".join(map(shown, table[category])))
        lines.append("")
    lines += [
        "Differences are calculated before rounding; display uses three decimals.",
        "Table II: equal configuration weights within category.",
        "Table III: equal user weights within category; Overall weights categories equally.",
    ]
    return "\n".join(lines) + "\n"


def csv_text(name: str, table: dict) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    headers = HEADERS[name]
    writer.writerow(["category", *headers, *(h + "_3dp" for h in headers)])
    for category in TABLE_ORDER[name]:
        values = table[category]
        writer.writerow([category, *map(str, values), *map(shown, values)])
    return stream.getvalue()


def validate_output_directory(path: Path, inputs: tuple[Path, Path]) -> Path:
    """Refuse overwrites, dangling symlinks, and known reference locations."""
    if path.exists() or path.is_symlink():
        raise ValueError("Output directory already exists; refusing overwrite")
    path = path.resolve()
    if not path.parent.is_dir():
        raise ValueError("Output directory's parent must already exist")
    protected = [(REPOSITORY_ROOT / "artifacts").resolve(), *(p.parent for p in inputs)]
    if any(path == root or root in path.parents for root in protected):
        raise ValueError("Output directory must be outside reference/input directories")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--factorial-csv", type=Path,
                        default=REPOSITORY_ROOT / FACTORIAL_RELATIVE)
    parser.add_argument("--matched-csv", type=Path,
                        default=REPOSITORY_ROOT / MATCHED_RELATIVE)
    parser.add_argument("--output-dir", type=Path,
                        help="optional NEW directory for the two CSV tables; otherwise stdout only")
    args = parser.parse_args(argv)
    try:
        input_paths = (args.factorial_csv.expanduser().resolve(strict=True),
                       args.matched_csv.expanduser().resolve(strict=True))
        destination = None
        if args.output_dir is not None:
            destination = validate_output_directory(args.output_dir.expanduser(), input_paths)
        data = {}
        for name, path in zip(EXPECTED_HASHES, input_paths):
            value = path.read_bytes()
            if sha(value) != EXPECTED_HASHES[name]:
                raise ValueError(f"{name}: SHA-256 does not match the accepted input")
            data[name] = value
        tables, coverage = calculate(data["factorial"], data["matched_stream"])
        # Build the complete output contents before creating anything.
        text = output_text(tables)
        files = {name.replace(":", "_") + ".csv": csv_text(name, table)
                 for name, table in tables.items()}
        if destination is not None:
            destination.mkdir(mode=0o700, exist_ok=False)
            for filename, content in files.items():
                with (destination / filename).open("x", encoding="utf-8", newline="") as handle:
                    handle.write(content)
        print("Input SHA-256 checks: MATCH")
        for name, value in coverage.items():
            print(f"{name}: {value}")
        print()
        print(text, end="")
        if destination is None:
            print("No output files written (use --output-dir to save the two CSVs).")
        else:
            print(f"Saved two table CSVs in: {destination}")
        return 0
    except (OSError, ValueError, KeyError, csv.Error, InvalidOperation) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

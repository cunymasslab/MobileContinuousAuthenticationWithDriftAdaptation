# M7 Reference Bundle: Packaging Notes

Status: sanitized working copy for private replication-package preparation. This is not a complete replication package or an authorization for public redistribution.

## Source and verification

- Source archive: `tifs_m7_reference_bundle.zip`.
- Source archive SHA-256: `14b340920855456d05361a4088e1042e517879d2f02ec631c356ef5a9faf9203`.
- All 11 expected source files were present; ZIP CRC checks passed.
- The four entries in the candidate manifest and the five entries in the statistical-output manifest were independently checked against the uploaded bytes. All nine matched.
- The repaired summary also matched its previously recorded accepted SHA-256: `7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098`.
- These are file-identity and packaging checks, not reruns of the scientific acceptance audits, training, statistics, or numerical tests.
- The uploaded source archive has not been changed.

## Changes in this derivative

Only private home-directory path prefixes in four text reports were replaced with `<COMPUTE_B_HOME>/`. The remaining experiment-directory components and artifact basenames were retained. Numerical text, coverage counts, outcomes, original input hashes, and supersession relationships were not changed.

| Text report | Path-prefix substitutions |
|---|---:|
| `MERGE_PROVENANCE.txt` | 4 |
| `summary_acceptance_audit.txt` | 3 |
| `detail_acceptance_audit.txt` | 6 |
| `m7_m8_statistical_closure/m7_m8_statistical_closure.txt` | 1 |

All five CSV files are byte-for-byte unchanged, including their participant identifiers, row order, missing values, and numerical precision. The two copied `MANIFEST.sha256` files were regenerated to describe this sanitized derivative. They must not be represented as the original historical manifests. The original manifests are reproduced below for provenance.

This file is new packaging documentation, not a September 1 research record.

## Original and packaged SHA-256 values

| Relative path | Original SHA-256 | Packaged SHA-256 |
|---|---|---|
| `results_stage4_merged_summary_m7_repaired_20260901.csv` | `7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098` | `7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098` |
| `MANIFEST.sha256` | `9ada81e496446f0e0045d4b51ae289d24d2dd2604c355f1bca1dc354ea0a0f9a` | `9ba4780ddbe61e2fcd9279f1ecd5df1ec70a3ca42d447fa787d1db4a4f1772c7` |
| `MERGE_PROVENANCE.txt` | `b49208e7cd295e439a073ce3a4f8ee9e79e25ac83f04eda7a8a7dd73338bebd6` | `58924be312e4f283209f40ecc452732decc8709eb63fbf5b1f282143864d412c` |
| `summary_acceptance_audit.txt` | `9c16cb210658b9384e4d6eb267a0c47e2bc6c76f650d58487299532d1b7edcd5` | `fb33990d0609d0a89800d4a77a74796d246ec07519ac0cf95e0d8db596eb98b5` |
| `detail_acceptance_audit.txt` | `eba1c92cfaced1d58f2c0104c60fe7240f043e6e88940749027e6d4ad62a2ba2` | `ec5e4118a2f8ba74b7ced9f92a2e0cad7c43a1bcbef7bb158f527bdfd14533e1` |
| `m7_m8_statistical_closure/MANIFEST.sha256` | `64b819d5501c97b7255ef645123e6a5a4cd1f5c4532156479990da1c4a535692` | `d9f3d0141a9726d4d8d8ff0319ab17e22ad185b56f57c5f25c2f37441d4f88ae` |
| `m7_m8_statistical_closure/m7_m8_statistical_closure.txt` | `3f5e5722f031990e2eae726affbd39fe88249a0bd4e0e1765f029b18ec9cafc8` | `b8c4e248f5432b23a439d2d6ad2d7151cb246b2fdebabc14e5bcee728665ca37` |
| `m7_m8_statistical_closure/m7_m8_overall_statistics.csv` | `1517e33ce01ffb253a2c58b51149d6daf98af2921929f09e38645822327ab7b1` | `1517e33ce01ffb253a2c58b51149d6daf98af2921929f09e38645822327ab7b1` |
| `m7_m8_statistical_closure/m7_m8_by_category.csv` | `3cd87286912bfa819a805e1611817bd1d55a186015878fafb362ab32581d045c` | `3cd87286912bfa819a805e1611817bd1d55a186015878fafb362ab32581d045c` |
| `m7_m8_statistical_closure/m7_m8_user_level_values.csv` | `5941ed798e83399a7d579680acf2b61edfc3f9596654ba15db0fddb4c9fc460f` | `5941ed798e83399a7d579680acf2b61edfc3f9596654ba15db0fddb4c9fc460f` |
| `m7_m8_statistical_closure/m7_m8_user_category_values.csv` | `23017203dfab7aff57b56481542a56fe21da87d6f36834e3795a980622049057` | `23017203dfab7aff57b56481542a56fe21da87d6f36834e3795a980622049057` |

## Original candidate manifest

```text
7217062926f755dbbb4259b20b49680a96d4647f1e3651861d52659f9cc3d098  results_stage4_merged_summary_m7_repaired_20260901.csv
b49208e7cd295e439a073ce3a4f8ee9e79e25ac83f04eda7a8a7dd73338bebd6  MERGE_PROVENANCE.txt
9c16cb210658b9384e4d6eb267a0c47e2bc6c76f650d58487299532d1b7edcd5  summary_acceptance_audit.txt
eba1c92cfaced1d58f2c0104c60fe7240f043e6e88940749027e6d4ad62a2ba2  detail_acceptance_audit.txt
```

## Original statistical-output manifest

```text
3f5e5722f031990e2eae726affbd39fe88249a0bd4e0e1765f029b18ec9cafc8  m7_m8_statistical_closure.txt
1517e33ce01ffb253a2c58b51149d6daf98af2921929f09e38645822327ab7b1  m7_m8_overall_statistics.csv
3cd87286912bfa819a805e1611817bd1d55a186015878fafb362ab32581d045c  m7_m8_by_category.csv
5941ed798e83399a7d579680acf2b61edfc3f9596654ba15db0fddb4c9fc460f  m7_m8_user_level_values.csv
23017203dfab7aff57b56481542a56fe21da87d6f36834e3795a980622049057  m7_m8_user_category_values.csv
```

## Scope and remaining boundaries

The text reports document prior scientific work. The detail acceptance report covers the two repaired detail chunks; the large detail inputs and any full-cohort detail artifact are not included. No executable analysis or training scripts were supplied in this reference bundle.

A bounded scan of the 11 packaged payload files found no remaining known private host labels, personal Unix/macOS home paths, root-user paths, IPv4-address shapes, user-at-host forms, private-key headers, or the checked token shapes. This is not a comprehensive security/privacy review. Dataset-participant IDs remain unchanged; data-use terms and redistribution permissions still require review before public release.

No files were uploaded to GitHub or to an agent workspace during creation of this archive.

## Layout

The archive contains this folder tree:

```text
artifacts/reference/m7_repaired_20260901/
  results_stage4_merged_summary_m7_repaired_20260901.csv
  MANIFEST.sha256
  MERGE_PROVENANCE.txt
  summary_acceptance_audit.txt
  detail_acceptance_audit.txt
  PACKAGING_NOTES.md
  m7_m8_statistical_closure/
    MANIFEST.sha256
    m7_m8_statistical_closure.txt
    m7_m8_overall_statistics.csv
    m7_m8_by_category.csv
    m7_m8_user_level_values.csv
    m7_m8_user_category_values.csv
```

Keep the original private source archive separately. Do not replace the original server reports or their saved manifests with these sanitized copies.

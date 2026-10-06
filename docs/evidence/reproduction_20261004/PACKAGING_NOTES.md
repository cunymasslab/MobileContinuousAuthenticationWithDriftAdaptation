# Packaging record for the verification-evidence update

## Source identity and checks

Uploaded archive: `tifs_verified_analysis_records.zip`.

SHA-256: `7fd8956e68f9301d51ac56266e0a2b94baa6bb5bd8cfb8dc816ce6eeca7b5377`.

The uploaded ZIP contains 21 entries: 19 selected files, collection notes, and a manifest. All 20 manifest entries match their uploaded bytes. Archive names are unique and relative with no parent traversal. The uploaded ZIP is retained unchanged as the maintainer's private archival source.

This preparation read and compared saved files, parsed Python source without executing it, and checked recorded package versions against the supplied installation metadata. It did not rerun any analysis, model, statistical fit, bootstrap, experiment, or installation. No GitHub repository was changed.

## Transformations

The private home-directory prefix was replaced by `<COMPUTE_B_HOME>` in five logs (38 occurrences total). The replacement was byte-oriented: no line-ending, numerical, timestamp, warning, or general whitespace cleanup was applied. The original collection manifest is preserved verbatim under `records/ORIGINAL_COLLECTION_MANIFEST.sha256`; its paths and hashes describe the original archive, not the sanitized copies. It was not silently regenerated under the original name.

Both code files, both matched-stream reports, their strict-comparison failure and diff, both M9 CSVs, all comparison records, both requirement lists, and the installation JSON remain byte-for-byte unchanged. Aliases at `analysis/reproduce_m9_comparison.py` and `requirements/reanalysis-python313.txt` are exact copies of the corresponding retained files. The manifest below, this packaging record, and the evidence README are newly prepared metadata.

The known private-home prefix and private host labels, home paths, user-at-host forms, and common credential shapes were scanned in this prepared bundle. An IP-shaped string inside third-party package metadata was inspected and was a public dependency version (`lz4`), not an infrastructure address. This is a scoped review of this bundle, not a complete repository/history privacy or security clearance. Public package metadata and license/author fields in the installation JSON were not removed.

## Original-to-packaged map

Destination paths below are relative to the repository root. All original entries are represented, including the original manifest and collection notes. A changed hash means only the documented prefix substitution, unless explicitly identified as new metadata.

| Original archive entry | Prepared repository path | Original SHA-256 | Prepared SHA-256 | Private-prefix replacements |
|---|---|---|---|---:|
| `COLLECTION_MANIFEST.sha256` | `docs/evidence/reproduction_20261004/records/ORIGINAL_COLLECTION_MANIFEST.sha256` | `7082bba683961b568e49e2c40b43ab24c857a143f2979eeb70769efc2a2f6e7d` | `7082bba683961b568e49e2c40b43ab24c857a143f2979eeb70769efc2a2f6e7d` | 0 |
| `COLLECTION_NOTES.txt` | `docs/evidence/reproduction_20261004/records/COLLECTION_NOTES.txt` | `47a00c60ff5db222b8f70b75588817bfbdf429e22a57b40fa47e7617e9c88437` | `47a00c60ff5db222b8f70b75588817bfbdf429e22a57b40fa47e7617e9c88437` | 0 |
| `environment/dependency-check.log` | `docs/evidence/reproduction_20261004/records/environment/dependency-check.log` | `9261363b733079a641c2e4cc9bc46ffa1d8336945a87f807b6cf68847dbc9b09` | `9261363b733079a641c2e4cc9bc46ffa1d8336945a87f807b6cf68847dbc9b09` | 0 |
| `environment/import-check.log` | `docs/evidence/reproduction_20261004/records/environment/import-check.log` | `1a53c8314b350a249de7804316f11707da667259ee7aaf3dfde48b3c799b09db` | `9c93a4ab49cdd89ab8036042ec877de9ed05d3d5c07031bcad7459a51910a7d7` | 1 |
| `environment/install-report.json` | `docs/evidence/reproduction_20261004/records/environment/install-report.json` | `969db366f152cdd1d22897713a3bf45a9119e44ac003ba4122da17b651fa6fd2` | `969db366f152cdd1d22897713a3bf45a9119e44ac003ba4122da17b651fa6fd2` | 0 |
| `environment/install.log` | `docs/evidence/reproduction_20261004/records/environment/install.log` | `066337d240b1ea640b110389790076db2ea061dc5aa34423bc03cd138802d631` | `75675c5629a1da55b4abc142127034af9ee353f667b3dfa05220266ebb76fe5b` | 17 |
| `environment/requirements-requested.txt` | `docs/evidence/reproduction_20261004/records/environment/requirements-requested.txt` | `96da51b1c92b2de0ea3bbc83b6c9e180998e3cc2a20986581c09018eaada702b` | `96da51b1c92b2de0ea3bbc83b6c9e180998e3cc2a20986581c09018eaada702b` | 0 |
| `environment/requirements-resolved.txt` | `docs/evidence/reproduction_20261004/records/environment/requirements-resolved.txt` | `55e7ccca819a122d49caeafa075108253a6d0dcad05517e01377920cbc95fda4` | `55e7ccca819a122d49caeafa075108253a6d0dcad05517e01377920cbc95fda4` | 0 |
| `m7/execution.log` | `docs/evidence/reproduction_20261004/records/m7/execution.log` | `96c15b7420778ae96f862891f457f4763e1acda57e7f03fece0ffccd1ca9310f` | `2e55b115a18c31f4e8f0ec8af956d57bd46165529966e48345a94aa54b01af6f` | 13 |
| `m7/output_comparison__1y0flmt.txt` | `docs/evidence/reproduction_20261004/records/m7/output_comparison__1y0flmt.txt` | `69a51a89cc707fe2de121dff8d9f1edaf4acf20a9df8eee1ac512fe52eebebeb` | `69a51a89cc707fe2de121dff8d9f1edaf4acf20a9df8eee1ac512fe52eebebeb` | 0 |
| `m9_sweep/execution.log` | `docs/evidence/reproduction_20261004/records/m9_sweep/execution.log` | `7767c44506b82e154d47ffdeae1a56cd8a37aefec54fd4fb9f972acb5fb53835` | `bebf8f1b8c05b92ff361883048203d140cd3e01ae859891b51b241b8c85c787f` | 1 |
| `m9_sweep/output_comparison.txt` | `docs/evidence/reproduction_20261004/records/m9_sweep/output_comparison.txt` | `06755644b3c0922cec5612cdfd0e6c60344420bb982996983f5cb49b5325868f` | `06755644b3c0922cec5612cdfd0e6c60344420bb982996983f5cb49b5325868f` | 0 |
| `m9_sweep/reference/comparison.csv` | `docs/evidence/reproduction_20261004/records/m9_sweep/reference/comparison.csv` | `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8` | `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8` | 0 |
| `m9_sweep/reproduce_m9_comparison.py` | `docs/evidence/reproduction_20261004/records/m9_sweep/reproduce_m9_comparison.py` | `50aadce3c5450182fa2c3e64371e0d1bb364b1cd13eb8fb941ed2edff7f5762a` | `50aadce3c5450182fa2c3e64371e0d1bb364b1cd13eb8fb941ed2edff7f5762a` | 0 |
| `m9_sweep/results_m9_lr_sweep/comparison.csv` | `docs/evidence/reproduction_20261004/records/m9_sweep/results_m9_lr_sweep/comparison.csv` | `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8` | `ba18ffaeb95ff02927dc039653c5edd78261e60851ebadd61e9a6316ab07dfd8` | 0 |
| `matched_stream/analyze_confound.py` | `docs/evidence/reproduction_20261004/records/matched_stream/analyze_confound.py` | `2d215f1813434c2784c7fb4efa36842d7c1bb6bf2dd2964d9204ef61cb0a15fe` | `2d215f1813434c2784c7fb4efa36842d7c1bb6bf2dd2964d9204ef61cb0a15fe` | 0 |
| `matched_stream/execution.log` | `docs/evidence/reproduction_20261004/records/matched_stream/execution.log` | `9d8a11f1ddb9eb3017810236da40dad19e39f8bd5dc05d896d4dcddda22be7e4` | `6ad17d6bcfdb5c00d71a4e3dcf0009061d4d936b8db90a24f2de02a0291032fc` | 6 |
| `matched_stream/reference/analysis_confound.txt` | `docs/evidence/reproduction_20261004/records/matched_stream/reference/analysis_confound.txt` | `90fb8051bbba1d027e2ec289553c28f8640dc5c7c2d5f0997a447bb967028295` | `90fb8051bbba1d027e2ec289553c28f8640dc5c7c2d5f0997a447bb967028295` | 0 |
| `matched_stream/report_comparison.txt` | `docs/evidence/reproduction_20261004/records/matched_stream/report_comparison.txt` | `b31044cb98e26822534a18a1d996265a95f38b3e518d2193874e33cf9ddaa098` | `b31044cb98e26822534a18a1d996265a95f38b3e518d2193874e33cf9ddaa098` | 0 |
| `matched_stream/report_difference.diff` | `docs/evidence/reproduction_20261004/records/matched_stream/report_difference.diff` | `7b5a216b89667022ab39f9b6995310c140afa03d0db8cf0593fba41f39be727a` | `7b5a216b89667022ab39f9b6995310c140afa03d0db8cf0593fba41f39be727a` | 0 |
| `matched_stream/results_confound/analysis_confound.txt` | `docs/evidence/reproduction_20261004/records/matched_stream/results_confound/analysis_confound.txt` | `b492ade6606ace40f0aa890a25de3426a0f007d2509db45aea034afd197a5104` | `b492ade6606ace40f0aa890a25de3426a0f007d2509db45aea034afd197a5104` | 0 |

## Unchanged aliases

| Added repository path | Original archive member | SHA-256 |
|---|---|---|
| `analysis/reproduce_m9_comparison.py` | `m9_sweep/reproduce_m9_comparison.py` | `50aadce3c5450182fa2c3e64371e0d1bb364b1cd13eb8fb941ed2edff7f5762a` |
| `requirements/reanalysis-python313.txt` | `environment/requirements-resolved.txt` | `55e7ccca819a122d49caeafa075108253a6d0dcad05517e01377920cbc95fda4` |

## Import boundary

This is an additive evidence/code/environment update. It does not replace `requirements.txt`, active experiment code, the M7 analysis, existing result artifacts, or existing top-level documentation. Use a new branch based on `main` and check that the planned paths do not already exist before importing. This preparation observed `main` at `5ba62da24182da771486f23c7ee18c546606807f`; that is context, not a reason to reset a later branch. Final status/link consolidation in the existing reproduction guide, paper-to-code map, and provenance ledger remains a distinct documentation task.

The provided manifest is located at `docs/evidence/reproduction_20261004/PACKAGED_MANIFEST.sha256`. Its entries are relative to the repository root and include all files in this prepared update except the manifest itself. The prepared update contains 26 files, of which 24 are under this evidence directory and two are the unchanged operational aliases. Nothing in this ZIP should overwrite a scientific input or previous execution record.

# Packaging notes

Prepared for an additive repository import. Baseline checked via the live GitHub
connector: `main` at `5cda80ad8d53777e7c458b8bfba9589b1471c636`.
At inspection, `analysis/publication_figures/` and
`docs/evidence/publication_rendering_20261004/` were not present. This package
contains only new paths beneath those two directories. It does not alter the
existing results, reference bundles, requirements, central documentation or
historical top-level plotting programs. No Git write was performed here.

## What is and is not changed

All five runnable reconstructed scripts exactly match the tested draft bytes.
The five original snapshots, both capture-helper snapshots, runner snapshot,
all numerical traces and title-layout records, declared source edits, original
manifests, and all fourteen selected PNG images are copied byte-for-byte.
The seven manuscript references are unchanged. No image was redrawn or used to
replace a reference in the comparison records.

Only a private server-home prefix was replaced with `<COMPUTE_B_HOME>` in
16 copied text records (72 occurrences).
This includes the two batch JSON reports and all fourteen execution logs. No
warning, numerical value, timestamp, status, relative suffix or traceback was
removed. The failing first-run states and all strict-pixel mismatches remain false.

| Packaged record | Prefix replacements | Original SHA-256 | Packaged SHA-256 |
|---|---:|---|---|
| `records/first/PUBLICATION_RENDER_CHECK.json` | 11 | `562c24afa1295a2f2929a7e0050bd9d864daf5005b1d272f5cdd42bd55023376` | `bd159ea34031c4e628f3e0c5d961a810747f0655ab3c68c6477c255830d3ead2` |
| `records/first/runs/drift_detection/original/execution.log` | 8 | `808649457d1cd244a9e8069945a2b77c81f9ed321624934bc29949b04a5921c8` | `74a34b887a53f1e677775dfc29a1918c10f0dc6b01a60c823600073cd90d5d2c` |
| `records/first/runs/drift_detection/draft/execution.log` | 8 | `b6da619cbca13b77fd57badc7a36ae762a345b63bff4f996adccb6f90b714e92` | `49fcbb2c99f841c679769f3b43a955c569c42ce1f61752191976509b2432504f` |
| `records/first/runs/drift_gap/original/execution.log` | 1 | `3c6ff1fb5530f220882bbd71778a3abbb322a9294fa3fd9355b7f3aad1a614fc` | `1fbf41725854d26c990c87c63b4b64a03c399763085d34f65467d19009891e69` |
| `records/first/runs/drift_gap/draft/execution.log` | 1 | `24048caf939836c89d0e8bfdb211d66d39fdea005e44e595967215e630eff6b0` | `14f2cdc42a1473f26a7f2864f109802ece0b8eff60c9c6a536ef0846b211ff9b` |
| `records/first/runs/forgetting/original/execution.log` | 1 | `9665b1abadddaae629994e2704eaf347bdf8a45e84f7e0db153dc47cd3811626` | `9e6b5a1f28838d61152dc472fe918f42964da486f5f96e03279dad09d4030103` |
| `records/first/runs/forgetting/draft/execution.log` | 1 | `77f443287ac407c0e6a661149dcc55fca28e94ab4a408f2644e054eaad46f7c2` | `47d9433db9bc109d6e060c72269e5c793f78c1c3a860660c1582c6116482dac3` |
| `records/first/runs/matched_stream/original/execution.log` | 9 | `05c4af648787ccb75a64adf8a29c43a7b6a4714fae0e4cd80028345ad100abba` | `32e48df158d1027e5ceaffc96414da19166f405ce31b445e1d492fde1eb12cb8` |
| `records/first/runs/matched_stream/draft/execution.log` | 9 | `662bb1c7c44cd903f25ca7281592febdc5a3f6f0b6b9c052e91126f4625158e0` | `ff1b961d54695e0a86836636c1ec7c4fd76707361d9d254ee606834e5bf82a9f` |
| `records/first/runs/m9_sweep/original/execution.log` | 1 | `dec1568914ec33a17ff020d31a10f20b49fe394e87eec3ab6a86a679ab066ac5` | `b27b30220ef0db1bc222c336314b4bc0017c8525eb5e72eddb05e1e1a85a77bc` |
| `records/first/runs/m9_sweep/draft/execution.log` | 1 | `d054f87082a32208e7a4e4f3726f62013367b587f0da63f88cfcab1889e593ec` | `99b297d1592acf071c59b03bd40a37a5965382cda299a321414db18be88f012a` |
| `records/retry/PUBLICATION_RENDER_CHECK.json` | 5 | `39775afc5c71aa701164e45ffbeeccffab3286a42e2bc6560a121c2648146ec4` | `0364b0af1e847e077d43064093dbf2afcd7c9d1de6814d828259d3a069086032` |
| `records/retry/runs/drift_detection/original/execution.log` | 4 | `c831e13804409790c0d46c8cbcd35baca058bfae3a0493bac31c7e2b6a8f47ec` | `926d4eb92894db1cb1f6d99e0f700aab011e34ed6a275554595d3c06a95179a9` |
| `records/retry/runs/drift_detection/draft/execution.log` | 4 | `69d182735a9697f1612f54ca5ca7f7223a7fe3ef894ff424d3988970eb45c2bc` | `d8fd61b2f49305ec72a881503f92d16e9b47fbbf390ca2298b3da5e2fae2faed` |
| `records/retry/runs/matched_stream/original/execution.log` | 4 | `59abab0fef011318cb8b27d780cfe1b2550c6f5954d8c5210c9596f85bfa5fad` | `c530a3c380e4827d67e097cd0898feb14d2ec681db20ad22faa0b2f93a11282d` |
| `records/retry/runs/matched_stream/draft/execution.log` | 4 | `7b91a05f4b0374b0f0440009d4248a8c490ae6acbee296abc62ba0150293f9a1` | `aa1345ed05feaefe60cb56cc50a30b9a96f533156ccc649c539bf544c6c3529e` |

[PACKAGING_INDEX.json](PACKAGING_INDEX.json) records every copied file's original
archive/path/hash and package path/hash, plus all excluded archive entries.
The two complete uploaded archives remain unchanged in the conversation and on
the maintainer's machines. Duplicated renderer snapshots, the seven original
colored image files, generated PDFs and supporting output images are omitted
from this curated addition. Every first/retry execution log and available trace
is preserved, so the retry does not hide the failure.

## Checks performed during packaging

Both ZIP hashes matched the maintainer's reported transfer values. All 72 first-run
and 57 retry output-manifest entries matched; manifest coverage was complete.
Stored renderer, reference and input identities were reconciled against the
previously supplied, checksum-verified test bundle. All five successful original
and draft trace files are byte-identical. Retained status records report exit 0,
zero changed staged files and zero missing outputs for the selected successful
pairs. All seven reference/draft PNG pairs were inspected without rerendering.
The full numerical-geometry comparison remains separate from the failed historical
pixel comparison; neither status is generalized beyond its scope.

Syntax parsing of the five runnable copies succeeded. Listed unchanged helper
function syntax trees in STATIC_CHECKS.json were compared again and match. The
new README's Bash example was parsed with `bash -n`; it was not executed. The
relocated direct invocation still belongs in the planned fresh-checkout test.

A limited text scan checked all packaged text for personal home prefixes, known
private hosts/addresses, user-at-host strings, private-key headers and common
credential-token forms after substitution. No matches remained. This scan is
not a full repository/history security audit or a participant-data license review.
No raw sensor data, CSV-input duplicates, fonts, private keys, caches, environment
binaries or full shell history are included.

## Manifest conventions

`records/first/ORIGINAL_OUTPUT_MANIFEST.sha256` and its retry counterpart are
unchanged full-archive identity records. They are **not** valid checksums for the
curated/sanitized package because some paths are relocated/omitted and log bytes
were sanitized. Input bundle manifests similarly refer to the original test inputs.
Use `PACKAGED_MANIFEST.sha256` from the repository root to verify all other files
in this addition; it does not include itself. Do not edit an original manifest
to erase the relationship to the unmodified archives.

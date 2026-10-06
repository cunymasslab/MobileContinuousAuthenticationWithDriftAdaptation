# RTRL Streaming Experiment Blueprint
**Version:** 3.0 | **Date:** March 31, 2026
**Target:** IEEE Transactions on Mobile Computing
**Researcher:** Avinash Jairam, PhD Candidate, CUNY Graduate Center

---

## 1. PROJECT OVERVIEW

### What This Experiment Proves
RTRL-based online adaptation recovers authentication performance under behavioral drift on mobile devices. A frozen model cannot adapt — its EER degrades as drift accumulates. An RTRL-adapting model updates weights per-timestep and recovers EER over time. This recovery curve is the core thesis contribution.

### Dataset
**HMOG Dataset** — 100 users, 24 sessions each
- 6 conditions: READING/WRITING/MAP × SITTING/WALKING
- Sensors: Accelerometer, Gyroscope, Magnetometer (3 sensors × 3 axes + magnitude = 12 features)
- Sampling rate: ~98 Hz
- 4 sessions per condition: 3 for training, 1 for streaming/testing

### Core Logic
1. Train RNN offline with BPTT on 3 training sessions of condition A
2. Clone trained weights into frozen and adapting variants
3. Stream session 4 data from condition A (temporal drift) or condition B (cross-condition drift)
4. Stream = interleaved legit + impostor windows, balanced 1:1, shuffled with per-user seed
5. Hidden state reset to zeros before each window (windows are independent)
6. RTRL model updates weights per-timestep during streaming; frozen model never updates
7. Measure EER over time — frozen stays high, adapting recovers

### Known Limitation
Ground truth labels are used during streaming (supervised setup). In real deployment, labels are unavailable. This is stated as future work in the paper.

---

## 2. LOCKED HYPERPARAMETERS

All values finalized through Stages 0-2. No further tuning.

| Parameter | Value | Source |
|-----------|-------|--------|
| Window size | 100 | Stage 0 |
| Step size | 100 | Stage 0 |
| Optimizer | Adam | Stage 1 |
| LR_train | 0.01 | Stage 1 |
| LR_online | 0.002 | Stage 2 (mean end EER=0.063, 2/57 forgetting) |
| Hidden dim (RTRL) | 32 | Fixed (O(H⁴) constraint) |
| Early stop patience | 3 | Fixed |
| Early stop delta | 1e-3 | Fixed |
| Max epochs | 30 | Fixed |
| Gradient clip | 1.0 | Fixed |
| Seeds | 42, 123, 456 | Fixed |
| Stream impostors | 3 | Fixed |
| Stream balance | 1:1 | Fixed |
| Rolling window | 50 | Fixed |

---

## 3. THE 8 MODELS

| ID | Name | Architecture | Training | Streaming Behavior | Purpose |
|----|------|-------------|----------|-------------------|---------|
| M1 | VanillaRNN-Frozen | VanillaRNN H=32 | NumPy BPTT | No updates | Primary baseline |
| M2 | VanillaRNN-RTRL | VanillaRNN H=32 | NumPy BPTT | RTRL per-timestep, all windows | Core contribution |
| M3 | VanillaRNN-RTRL-LegitOnly | VanillaRNN H=32 | NumPy BPTT | RTRL per-timestep, legit only | Ablation |
| M4 | VanillaRNN-BPTTFinetune | VanillaRNN H=32 | NumPy BPTT | BPTT per window, Adam | Justifies RTRL vs fine-tuning |
| M5 | LSTM-Frozen | LSTM H=32 | PyTorch | No updates | Capacity argument |
| M6 | GRU-Frozen | GRU H=32 | PyTorch | No updates | Architecture comparison |
| M7 | GRU-RTRL | GRU H=32 | PyTorch → NumPy transfer | RTRL per-timestep, all windows | RTRL generalizes to gated architectures |
| M8 | GRU-BPTTFinetune | GRU H=32 | PyTorch | BPTT per window, Adam | GRU fine-tune baseline |

### Shared Training
- M1-M4 share one VanillaRNN trained via NumPy BPTT with early stopping
- M5 uses a separately trained PyTorch LSTM
- M6, M7, M8 share one GRU trained via PyTorch with early stopping
- M7 transfers PyTorch GRU weights to NumPy GRU via `load_from_pytorch`

### GRU Weight Transfer (M7)
PyTorch GRU has two separate candidate biases (`b_n_input`, `b_n_hidden`). Reset/update gate biases can be summed, but candidate biases must NOT be summed — they are kept separate to match PyTorch's convention: `n_t = tanh(W_xn @ x_t + b_n_input + r_t * (W_hn @ h_prev + b_n_hidden))`.

### Forgetting Baseline Map
- M2, M3, M4 compared against M1 (VanillaRNN-Frozen)
- M7, M8 compared against M6 (GRU-Frozen)
- Forgetting flag: adapt_end_eer > frozen_end_eer + 0.05

---

## 4. COMPLETED STAGES

### Stage 0 — Preprocessing Search ✓
- Grid: Window {100, 200, 400} × Step {100, 200}
- Winner: Window=100, Step=100
- File: `results_stage0_preprocessing.csv`

### Stage 1 — Offline Training Hyperparameters ✓
- Grid: {Adam, SGD+momentum} × LR {0.0001, 0.001, 0.01}
- Winner: Adam, LR=0.01
- File: `results_stage1_training_hparams.csv`

### Stage 2 — Online Learning Rate ✓
- Grid: LR_online {0.0001, 0.0005, 0.001, 0.002}
- Winner: LR_online=0.002 (mean end EER=0.063, 2/57 forgetting cases)
- File: `results_stage2_online_hparams.csv`

### Stage 3 — Architecture Search ✓
- Grid: {VanillaRNN, LSTM, GRU} × H={32, 64, 128}, frozen only
- Finding: Larger frozen models do not match RTRL-adapting VanillaRNN at H=32
- File: `results_stage3_architecture_search.csv`

### Stage 4 — Full Main Experiment (M1-M6, M8) ✓
- 89/90 users completed (99% coverage), 12 parallel chunks
- 9,558 total experiments (89 users × 36 pairs × 3 seeds)
- Files: `results_stage4_merged_summary.csv`, `results_stage4_merged_detail.csv`

### M7 Validation Gates ✓
- Gate 0 (weight transfer): PASS (diff=2.42e-08)
- Gate 1 (H=1 numerical): PASS (max diff ~1e-10)
- Gate 2 (H=4 numerical): PASS (max diff ~1e-11)
- Gate 3 (RTRL=BPTT offline): PASS (max diff ~5e-17)
- Gate 4 (PyTorch transfer + numerical at H=32): PASS (max diff ~1e-11)

### Stage 4 — M7 Runs (IN PROGRESS)
- Linode (4 cores): chunks 1-2 running
- Tandem (12 cores): chunks 3-8 running
- Chunks 9-12: queued for when tandem frees up
- Estimated completion: mid-April 2026

---

## 5. STAGE 4 RESULTS (M1-M6, M8)

### Overall Mean EER by Model

| Model | Mean EER | Median EER | Std |
|-------|----------|------------|-----|
| M1 (VanillaRNN-Frozen) | 0.3145 | 0.2721 | 0.2451 |
| M2 (VanillaRNN-RTRL) | 0.1448 | 0.1172 | 0.1255 |
| M3 (VanillaRNN-RTRL-LegitOnly) | 0.2224 | 0.2073 | 0.1580 |
| M4 (VanillaRNN-BPTTFinetune) | 0.1450 | 0.1171 | 0.1255 |
| M5 (LSTM-Frozen) | 0.2950 | 0.2435 | 0.2440 |
| M6 (GRU-Frozen) | 0.2953 | 0.2457 | 0.2431 |
| M8 (GRU-BPTTFinetune) | 0.0906 | 0.0607 | 0.0956 |

### Mean EER by Model × Drift Category

| Model | Temporal | Cross-Posture | Cross-Activity | Cross-Both |
|-------|----------|---------------|----------------|------------|
| M1 | 0.2212 | 0.3235 | 0.3028 | 0.3685 |
| M2 | 0.1026 | 0.1546 | 0.1342 | 0.1717 |
| M3 | 0.1584 | 0.2328 | 0.2118 | 0.2599 |
| M4 | 0.1026 | 0.1548 | 0.1342 | 0.1721 |
| M5 | 0.2021 | 0.3028 | 0.2791 | 0.3533 |
| M6 | 0.1972 | 0.3107 | 0.2747 | 0.3573 |
| M8 | 0.0571 | 0.0987 | 0.0814 | 0.1124 |

### Key Findings
1. **RTRL cuts EER by 54%:** M2 (0.1448) vs M1 (0.3145), winning 89.4% of experiments
2. **RTRL = BPTT fine-tuning for VanillaRNN:** M2 (0.1448) ≈ M4 (0.1450)
3. **Legit-only adaptation is weaker:** M3 (0.2224) — updating on all windows is better
4. **Bigger drift = bigger RTRL benefit:** Temporal +0.12, cross-both +0.20 EER reduction
5. **Frozen architectures are equivalent:** M1 (0.31), M5 (0.30), M6 (0.30) — architecture alone doesn't help
6. **GRU-BPTT is the best so far:** M8 (0.0906), but M7 may beat it
7. **Forgetting is rare:** Only 9.4% of experiments flagged
8. **Mean recovery:** M2: 0.66, M8: 0.86 (median recovery for M8: 1.0 = full recovery)

---

## 6. PRELIMINARY M7 RESULTS (23 users, in progress)

| Server | Users | M6 (Frozen) | M7 (RTRL) | Win Rate | Recovery Median |
|--------|-------|-------------|-----------|----------|-----------------|
| Linode | 6 | 0.2162 | 0.0546 | 92.9% | 1.00 |
| Tandem | 17 | 0.2588 | 0.0773 | 87.2% | 1.00 |

### M7 by Drift Category (Linode, 6 users)

| Category | M6 (Frozen) | M7 (RTRL) | Improvement |
|----------|-------------|-----------|-------------|
| Temporal | 0.1424 | 0.0355 | 0.1069 |
| Cross-Posture | 0.2234 | 0.0567 | 0.1667 |
| Cross-Activity | 0.2002 | 0.0513 | 0.1490 |
| Cross-Both | 0.2659 | 0.0665 | 0.1993 |

### Significance
- M7 (GRU-RTRL: ~0.05-0.08) is tracking **below** M8 (GRU-BPTT: 0.09)
- Unlike VanillaRNN where RTRL merely matched BPTT, GRU-RTRL appears to **outperform** GRU-BPTT
- This is because RTRL updates per-timestep (finer granularity) vs BPTT updating per-window
- If confirmed across all 90 users, this directly answers the reviewer question: "Why RTRL over BPTT?"

---

## 7. GRU-RTRL IMPLEMENTATION DETAILS (M7)

### Architecture
- 12 parameters: W_xr, W_hr, b_r (reset), W_xz, W_hz, b_z (update), W_xn, W_hn, b_n_input, b_n_hidden (candidate), W_hy, b_y (output)
- PyTorch GRU convention: reset gate applied AFTER matrix multiply: `r_t * (W_hn @ h_prev + b_n_hidden)`
- 10 sensitivity tensors tracked per timestep (all recurrent parameters)

### RTRL Sensitivity Update
For each parameter θ, the sensitivity S[θ] = dh/dθ is updated via:

```
S_new[θ] = dz/dθ * (h_prev - n_t) + (1-z_t) * dn/dθ + z_t * S_prev[θ]
```

Where the candidate derivative uses the product rule for `r_t * q_t`:
```
d(r*q)/dθ = dr/dθ * q + r * dq/dθ
```

### Complexity
- Time: O(H⁴) per timestep (10 sensitivity matrices of size H×H×D or H×H×H)
- Memory: O(H³) (sensitivity tensors)
- At H=32: ~1M operations per timestep, ~1000x slower than VanillaRNN RTRL

### Files
- `gru_model.py` — GRU_NumPy class with RTRL
- `test_gru_forward.py` — Gate 0 (weight transfer validation)
- `test_gru_rtrl.py` — Gates 1-4 (numerical + offline equivalence)

---

## 8. STAGE 5 — SnAp-n SPARSITY (NOT YET STARTED)

**Prerequisites:** Stage 4 complete (including M7). All GRU+RTRL gates passed ✓.

### Goal
Show RTRL scales beyond H=32 using sparse Jacobian approximation.

### Models

| Model | H | Model Sparsity | SnAp Level | Purpose |
|-------|---|---------------|-----------|---------|
| M9 | 32 | 0% | SnAp-1 | SnAp-1 baseline at H=32 |
| M10 | 32 | 0% | SnAp-2 | Sanity check (must match M2) |
| M11 | 64 | 0% | SnAp-1 | Larger H with SnAp-1 |
| M12 | 64 | 68% | SnAp-1 | Sparse model + SnAp-1 |
| M13 | 64 | 68% | SnAp-2 | Sparse model + SnAp-2 (lower bias) |

### SnAp-n Core Idea
Parameter θ_j only immediately influences its directly connected hidden unit. SnAp-n zeros out Jacobian entries for influences that cannot reach a hidden unit in n steps. This reduces O(H⁴) to O(d·H²) for SnAp-1 or O(d³·H⁴) for SnAp-2, where d = density.

### Key Constraints
- SnAp-2 on a dense network = full RTRL (no benefit without model sparsity)
- SnAp-1 has high approximation bias but BPTT-level compute cost
- Offline training still uses standard BPTT — SnAp-n is streaming-phase only
- Sensitivity matrices should NOT be reset between windows (preserves cross-window gradient information)

### Validation Gates (5 required)
1. SnAp-1 at H=1: must match full RTRL (tolerance 1e-4)
2. SnAp-2 at H=4 dense: must match full RTRL (tolerance 1e-4)
3. Sparsity enforcement: pruned weights must be exactly 0.0 after 100 updates
4. M10 sanity: must match M2 results (difference < 0.001 mean EER)
5. Offline equivalence: SnAp-2 dense weights must match full RTRL within 1e-5

---

## 9. METRICS

### Per-Experiment Metrics
- **EER** (Equal Error Rate) — primary metric, 1001-threshold sweep
- Accuracy, Precision, Recall, F1, MCC
- AUC-ROC (via sklearn)
- overall_eer, initial_eer (first 50 windows), end_eer (last 50 windows)
- recovery = (initial_eer - end_eer) / initial_eer
- Cumulative EER and rolling EER logged per window in detail CSV
- Forgetting flag per experiment

### Statistical Tests (Required for Paper)
- Paired t-test and Wilcoxon signed-rank test
- Mean EER improvement with 95% confidence interval
- Win rate (% of experiments where adapting beats frozen)
- Cohen's d (effect size)

---

## 10. OUTPUT FILES

### Stage 4 Results (on tandem server)
```
results_stage4_chunk{1-12}_summary.csv      — per-chunk results (M1-M6, M8)
results_stage4_chunk{1-12}_detail.csv        — per-window predictions
results_stage4_chunk{1-12}_training_curves.csv
results_stage4_merged_summary.csv            — merged (9,558 rows, 89 users)
results_stage4_merged_detail.csv             — merged (~8.7M rows)
```

### M7 Results (separate files, to be merged later)
```
results_stage4_m7_chunk{1-2}_summary.csv     — Linode
results_stage4_m7_chunk{3-8}_summary.csv     — Tandem
results_stage4_m7_chunk{9-12}_summary.csv    — Queued
```

### Merging M7 into Stage 4
After M7 completes, merge M7 columns into existing Stage 4 CSVs by matching on (target_user, train_condition, stream_condition, seed) key.

---

## 11. FIGURES REQUIRED FOR PAPER

1. **Recovery curve** — Rolling EER over window index for frozen vs adapting models. Core figure.
2. **Win rate by drift category** — Bar chart showing adaptation benefit scales with drift severity.
3. **Model comparison table** — 8 models × 4 drift categories, mean EER ± std (Table 1).
4. **Frozen capacity comparison** — Stage 3: frozen {VanillaRNN, LSTM, GRU} at H={32,64,128} vs RTRL H=32 line.
5. **GRU-RTRL vs GRU-BPTT** — M7 vs M8, demonstrating per-timestep granularity advantage.

---

## 12. CODEBASE

### Core Files
```
model.py              — VanillaRNN with BPTT and RTRL (M1-M4)
gru_model.py          — GRU_NumPy with RTRL (M7)
pytorch_models.py     — PyTorch LSTM/GRU/RNN classifiers (M5, M6, M8)
run_stage4.py         — Stage 4 experiment runner (all 8 models)
run_stage1.py         — AdamState optimizer, training with early stopping
run_streaming.py      — Legacy streaming (Stage 2 protocol)
```

### Data Pipeline
```
data_loader.py        — Load raw sensor CSVs per user/session
preprocessor.py       — compute_magnitude, segment_into_windows, normalize, create_dataset
run_experiment.py     — compute_eer, evaluate, confusion_matrix
```

### Utilities
```
merge_results.py      — Merge chunk CSVs with deduplication
split_users.py        — Validation/test user split
sanity_check.py       — Pre-launch assertions
test_gru_forward.py   — M7 Gate 0 (weight transfer)
test_gru_rtrl.py      — M7 Gates 1-4 (numerical + equivalence)
```

---

## 13. DESIGN DECISIONS TO DOCUMENT IN PAPER

1. **Hidden state reset per window:** Windows are independent — no hidden state carry-over. Justified because interleaved impostor windows break temporal continuity.

2. **Sensitivity matrices reset per window:** RTRL sensitivities track gradients within a single window, not across windows. This makes M2/M7 directly comparable to M4/M8 (all see one window at a time). True cross-window RTRL is future work.

3. **12 input features:** 3 sensors × (3 axes + 1 magnitude) = 12. Magnetometer included despite original HMOG paper finding it weaker — the RNN learns to weight channels appropriately.

4. **Sample-by-sample training:** Both NumPy and PyTorch models train one sample at a time (not mini-batch), ensuring consistent Adam dynamics across implementations.

5. **Supervised streaming:** Labels used during adaptation. Deployment-grade label acquisition is future work.

---

## 14. NEXT STEPS (In Order)

1. **Complete M7 runs** — Wait for all 12 chunks to finish (~mid-April)
2. **Merge M7 results** — Combine M7 columns into Stage 4 merged CSVs
3. **Full statistical analysis** — Paired tests, confidence intervals, effect sizes for all model comparisons
4. **Generate paper figures** — Recovery curves, comparison tables, drift category analysis
5. **Stage 5 (SnAp-n)** — Implement and validate, then run on 90 users
6. **Write paper** — Methods, results, discussion sections

---

## 15. SERVER STATUS (March 31, 2026)

### Tandem (12 cores)
- Stage 4 original (M1-M6, M8): COMPLETE (89/90 users)
- M7 chunks 3-8: RUNNING (~27% complete)
- Backup: `stage4_backup/` directory

### Linode (4 cores)
- M7 chunks 1-2: RUNNING (~29% complete)

### Queued
- M7 chunks 9-12: Launch on tandem when chunks 3-8 finish

---

*End of Blueprint v3.0*
*Last updated: March 31, 2026*

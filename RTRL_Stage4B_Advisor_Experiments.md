# RTRL Experiment — Stage 4B: Advisor-Requested Experiments
**Date:** April 8, 2026
**Context:** Post-advisor meeting, pre-TIFS submission
**Prerequisites:** M7 runs complete (~April 10-12), Stage 4 results merged
**Target Venue:** IEEE TIFS (changed from TMC per advisor discussion)

---

## OVERVIEW

These experiments address specific questions raised by the advisor on April 8, 2026.
They run AFTER M7 completes but BEFORE paper writing. Stage 5 (SnAp-n) is deferred
to a later chapter/paper.

---

## EXPERIMENT 1: Does More Training Data Eliminate Drift?

### Motivation
Advisor question: "Is RTRL just compensating for insufficient training data? If we
give the frozen model more data, does it match RTRL without any adaptation?"

If yes → drift is a data quantity problem, not an adaptation problem.
If no → drift is real, more data doesn't help, and adaptation is genuinely needed.

### Design

**Experiment 1A — Add streaming data to training:**
- Current protocol: train on sessions 1-3, stream all of session 4
- New protocol: train on sessions 1-3 + first 50% of session 4 windows, stream remaining 50%
- Run frozen models only (M1, M6) — no RTRL needed
- Compare frozen EER with 3 sessions vs 3.5 sessions of training data
- If frozen EER drops significantly → more data helps, weakens RTRL argument
- If frozen EER stays high → drift is real, RTRL is justified

**Experiment 1B — Reduce streaming to 50% of session 4:**
- Same training (sessions 1-3), but predict only second half of session 4
- This isolates whether the RTRL benefit comes from adapting on the first half
- Compare: frozen on full session 4 vs frozen on second half only

### Implementation
- Modify `run_stage4.py` to accept a `--train-stream-split` flag (default: 0.0 = current behavior)
- When set to 0.5: first 50% of session 4 windows added to training set, remaining 50% used for streaming
- Run on all 89 test users, 36 pairs, 3 seeds — frozen models only (fast, no RTRL)
- Output: `results_stage4b_exp1_summary.csv`

### Expected Runtime
- Frozen models only → ~1-2 days on tandem (12 cores, 6 chunks)

---

## EXPERIMENT 2: Per-Session Cross-Validation (Drift Verification)

### Motivation
Advisor question: "Show that the model works within the training distribution but
degrades on session 4 — proving drift exists, not just poor training."

### Design
- For sessions 1-3 of each condition: use 2/3 for training, 1/3 for testing
- Compute frozen model EER on session 1-3 test splits (within-distribution performance)
- Compare against frozen model EER on session 4 (out-of-distribution / drift)
- Large gap between within-distribution and session-4 EER proves drift is real

### Implementation
- New script `run_stage4b_cv.py`
- For each user × condition × seed:
  - Split sessions 1-3: 2/3 train, 1/3 test
  - Train frozen model on 2/3
  - Evaluate on 1/3 (within-distribution EER)
  - Evaluate on session 4 (drift EER)
- Output: `results_stage4b_cv_summary.csv` with columns:
  `target_user, condition, seed, within_eer, session4_eer, drift_gap`

### Expected Runtime
- Frozen models only → ~1 day on tandem

---

## EXPERIMENT 3: Drift Visualization

### Motivation
Advisor: "Show the shift detection algorithm. In the paper we want to show the shift."

### Design — Three visualizations:

**Viz 3A — Rolling EER over time (frozen vs adapting):**
- Plot rolling EER (window 50) over window index for M1 (frozen) vs M2 (RTRL)
- Average across all users and conditions, with confidence bands
- Frozen line stays high / rises; RTRL line drops — this IS the drift + recovery story
- Already have data in `results_stage4_merged_detail.csv`

**Viz 3B — Sensor distribution shift:**
- For a representative user, plot the distribution of accelerometer magnitude
  for sessions 1-3 (training) vs session 4 (streaming) for cross-condition pairs
- Histogram or KDE overlay showing the distributional shift
- Shows drift is visible in raw sensor data, not just in model performance

**Viz 3C — EER heatmap by condition pair:**
- 6×6 heatmap: rows = train condition, columns = stream condition
- Cell value = mean frozen EER
- Diagonal (temporal) should be lighter; off-diagonal (cross-condition) darker
- Visual proof that cross-condition drift is worse

### Implementation
- Analysis script: `analyze_stage4b.py`
- Uses existing merged detail and summary CSVs — no new experiments needed

### Expected Runtime
- Analysis only — minutes

---

## EXPERIMENT 4: Semi-Supervised Adaptation (Pseudo-Labels)

### Motivation
TIFS reviewer concern: "Labels aren't available at inference time."

### Design
- New streaming behavior: adapt only when model confidence is high
- High confidence = score > 0.9 (predict legit) or score < 0.1 (predict impostor)
- Use the model's own prediction as the pseudo-label
- Two new models:
  - M2-SS: VanillaRNN-RTRL with pseudo-labels
  - M7-SS: GRU-RTRL with pseudo-labels

### Implementation
- Add new model type `rtrl_adapt_semisupervised` in `run_stage4.py`
- In streaming loop: compute score first, only call RTRL update if score > 0.9 or < 0.1
- Pseudo-label: 1 if score > 0.9, 0 if score < 0.1
- Run with `--models M1,M2,M2SS,M6,M7,M7SS` (need frozen baselines for comparison)

### Expected Runtime
- VanillaRNN models (M2-SS): ~3-4 days on tandem
- GRU models (M7-SS): ~2-3 weeks on tandem (same O(H⁴) cost as M7)
- Consider running M2-SS first as proof of concept

---

## EXPERIMENT 5: Sliding-Window Retrain Baseline

### Motivation
TIFS reviewer concern: "Why not just periodically retrain with recent data?"

### Design
- New model: M_retrain
- Every N windows (e.g., N=10), collect the last N windows + labels
- Retrain the model from current weights using BPTT on those N windows (1 epoch)
- This is what a practitioner might actually do — batch retrain periodically

### Implementation
- Add new model type `periodic_retrain` in `run_stage4.py`
- Parameter: retrain interval N (try N=10, N=25)
- Uses existing BPTT backward pass, no new gradient code
- Run for VanillaRNN and GRU

### Expected Runtime
- Similar to M4/M8 (BPTT fine-tune) — ~3-5 days on tandem

---

## MODEL SIZE AND COMPUTATIONAL COST TABLE

### For the paper (advisor requested):

| Model | Params | Weight Mem | Adaptation Overhead | Total Memory | Latency/Window | FLOPs/Timestep |
|-------|--------|-----------|--------------------:|-------------:|---------------:|---------------:|
| M1 VanillaRNN-Frozen | 1,473 | 11.5 KB | — | 11.5 KB | 0.4 ms | O(H²) |
| M2 VanillaRNN-RTRL | 1,473 | 11.5 KB | 360 KB | 371.5 KB | 49 ms | O(H⁴) |
| M4 VanillaRNN-BPTT-FT | 1,473 | 11.5 KB | 50 KB | 61.5 KB | 1.5 ms | O(H²) |
| M5 LSTM-Frozen | 5,921 | 46.3 KB | — | 46.3 KB | ~2 ms | O(H²) |
| M6 GRU-Frozen | 4,385 | 34.3 KB | — | 34.3 KB | 2.7 ms | O(H²) |
| M7 GRU-RTRL | 4,385 | 34.3 KB | 1,088 KB | 1,122 KB | 541 ms | O(H⁴) |
| M8 GRU-BPTT-FT | 4,385 | 34.3 KB | 100 KB | 134.3 KB | ~3 ms | O(H²) |

### Why RTRL is slower at the same model size:
- BPTT: O(H²) forward + O(H²) backward = O(H²) per timestep
- RTRL: O(H⁴) per timestep (maintaining H×H sensitivity matrices, multiplied by H×H weight matrices)
- VanillaRNN RTRL: 3 sensitivity tensors → 512× slower than BPTT
- GRU RTRL: 10 sensitivity tensors → 1,707× slower than BPTT
- Trade-off: RTRL uses more compute but O(1) memory in sequence length (no stored activations)

### RTRL vs BPTT Memory Trade-off:
| | RTRL | BPTT |
|---|---|---|
| **Memory** | O(H³) — sensitivity matrices, independent of T | O(T·H) — stored activations, grows with sequence |
| **Compute** | O(H⁴) per timestep | O(H²) per timestep |
| **Backward pass** | Not needed | Required |
| **Online capability** | True online — update at any timestep | Must complete full sequence first |

---

## PAPER VALUE STATEMENT

### Core Message
"Behavioral drift in mobile biometric authentication is a real, measurable, and
unsolved problem. We demonstrate that forward-mode gradient methods (RTRL) enable
continuous model adaptation that matches backward-mode methods (BPTT fine-tuning)
in quality, while offering a fundamentally different computational trade-off:
no stored activations, no backward pass, true per-timestep updates."

### Key Observations (for paper)
1. Frozen models degrade under drift regardless of architecture (M1≈M5≈M6 at ~0.30 EER)
2. Online adaptation reduces EER by 50-70% across all drift categories
3. RTRL matches BPTT fine-tuning on both VanillaRNN (M2≈M4) and GRU (M7≈M8)
4. Adaptation benefit scales with drift severity (temporal < cross-posture < cross-activity < cross-both)
5. GRU adaptation (0.09) outperforms VanillaRNN adaptation (0.14) — architecture matters when combined with adaptation
6. Forgetting is rare (9.4%) — online adaptation is safe
7. All models fit within mobile latency budget (≤541ms per window vs ~1s arrival rate)

### What This Paper is NOT Claiming
- Not claiming RTRL is faster than BPTT (it's slower)
- Not claiming this is deployment-ready (supervised labels)
- Not claiming novelty in RTRL algorithm itself (Williams & Zipser 1989)
- Claiming: first application of RTRL to biometric drift adaptation, with rigorous factorial comparison

---

## EXECUTION ORDER

1. **Wait for M7 to finish** (~April 10-12)
2. **Merge M7 results** into Stage 4 CSVs
3. **Experiment 1** (more training data) — 1 day code + 1-2 days run
4. **Experiment 2** (cross-validation drift proof) — 1 day code + 1 day run
5. **Experiment 3** (drift visualizations) — analysis only, concurrent with above
6. **Experiment 5** (sliding-window retrain) — 1 day code + 3-5 days run
7. **Experiment 4** (semi-supervised M2-SS first) — 1 day code + 3-4 days run
8. **Full statistical analysis** on all models
9. **Paper writing**

Experiments 1, 2 can run in parallel. Experiment 3 needs no compute.
Experiments 4, 5 can run in parallel after 1-2 finish.

**Estimated total timeline: 3-4 weeks from M7 completion to paper draft.**

---

## STAGE 5 (SnAp-n) — DEFERRED

SnAp-n sparsity experiments (M9-M13) are deferred to a later dissertation chapter.
The current paper focuses on full RTRL at H=32. Scalability to larger H via SnAp-n
will be mentioned as future work in the paper.

See Blueprint v3.0 Section 8 for the full SnAp-n plan when ready.

---

*End of Stage 4B Plan*
*This supplements RTRL_Blueprint_v3.md — all prior locked hyperparameters and
protocols remain unchanged.*

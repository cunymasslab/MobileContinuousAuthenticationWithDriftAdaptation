"""
test_rtrl_persistent.py

Unit tests for VanillaRNN.rtrl_forward_persistent.

Verifies:
  (a) Zero-state equivalence: rtrl_forward_persistent(X, 0, 0, 0, 0) matches
      rtrl_forward(X) bit-for-bit.
  (b) State threading: two consecutive calls with persistent state produce the
      same final state and sensitivities as one call on the concatenated sequence.
  (c) Horizon fix: gradient on window 2 with persistent state differs from
      gradient with reset state. Proves the fix changes observable behavior.

Run:
    python tests/test_rtrl_persistent.py
"""

import os
import sys

# Allow `from model import VanillaRNN` when running from repo root or tests/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from model import VanillaRNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rnn(H=4, D=3, seed=0):
    np.random.seed(seed)
    return VanillaRNN(input_dim=D, hidden_dim=H, output_dim=1)


def zero_state(H, D):
    return (
        np.zeros((H, H, D)),   # S_xh
        np.zeros((H, H, H)),   # S_hh
        np.zeros((H, H)),      # S_bh
        np.zeros(H),           # h
    )


# ---------------------------------------------------------------------------
# Test (a): zero-state equivalence with the old rtrl_forward
# ---------------------------------------------------------------------------

def test_a_zero_state_equivalence():
    H, D, T = 4, 3, 10
    rnn = make_rnn(H, D, seed=0)

    np.random.seed(42)
    X = np.random.randn(T, D)

    # Old method: resets internally
    h_old, Sxh_old, Shh_old, Sbh_old = rnn.rtrl_forward(X)

    # New method: explicit zero initial state
    Sxh0, Shh0, Sbh0, h0 = zero_state(H, D)
    h_new, Sxh_new, Shh_new, Sbh_new = rnn.rtrl_forward_persistent(
        X, Sxh0, Shh0, Sbh0, h0
    )

    # Both paths execute identical arithmetic on identical inputs; expect exact equality.
    assert np.array_equal(h_old, h_new), (
        f"h mismatch: max diff {np.max(np.abs(h_old - h_new))}"
    )
    assert np.array_equal(Sxh_old, Sxh_new), (
        f"S_xh mismatch: max diff {np.max(np.abs(Sxh_old - Sxh_new))}"
    )
    assert np.array_equal(Shh_old, Shh_new), (
        f"S_hh mismatch: max diff {np.max(np.abs(Shh_old - Shh_new))}"
    )
    assert np.array_equal(Sbh_old, Sbh_new), (
        f"S_bh mismatch: max diff {np.max(np.abs(Sbh_old - Sbh_new))}"
    )

    print("test_a_zero_state_equivalence: PASS (bit-identical to rtrl_forward)")


# ---------------------------------------------------------------------------
# Test (b): state threading matches one long call
# ---------------------------------------------------------------------------

def test_b_state_threading():
    H, D, T = 4, 3, 10
    rnn = make_rnn(H, D, seed=1)

    np.random.seed(43)
    X = np.random.randn(T, D)
    X1, X2 = X[:5], X[5:]

    Sxh0, Shh0, Sbh0, h0 = zero_state(H, D)

    # Path 1: one call over the full sequence
    h_full, Sxh_full, Shh_full, Sbh_full = rnn.rtrl_forward_persistent(
        X, Sxh0, Shh0, Sbh0, h0
    )

    # Path 2: two calls with state threading
    h1, Sxh1, Shh1, Sbh1 = rnn.rtrl_forward_persistent(
        X1, Sxh0, Shh0, Sbh0, h0
    )
    h2, Sxh2, Shh2, Sbh2 = rnn.rtrl_forward_persistent(
        X2, Sxh1, Shh1, Sbh1, h1
    )

    # Mathematically identical recursion applied to the same 10 samples in the same
    # order, so both paths should agree to floating-point tolerance.
    tol = 1e-12
    assert np.allclose(h_full, h2, atol=tol), (
        f"h: max diff {np.max(np.abs(h_full - h2))}"
    )
    assert np.allclose(Sxh_full, Sxh2, atol=tol), (
        f"S_xh: max diff {np.max(np.abs(Sxh_full - Sxh2))}"
    )
    assert np.allclose(Shh_full, Shh2, atol=tol), (
        f"S_hh: max diff {np.max(np.abs(Shh_full - Shh2))}"
    )
    assert np.allclose(Sbh_full, Sbh2, atol=tol), (
        f"S_bh: max diff {np.max(np.abs(Sbh_full - Sbh2))}"
    )

    print("test_b_state_threading: PASS (threaded 5+5 == one call of 10)")


# ---------------------------------------------------------------------------
# Test (c): gradient differs between persistent and reset on window 2
# ---------------------------------------------------------------------------

def test_c_gradient_differs_on_window_2():
    H, D, T = 4, 3, 10
    rnn = make_rnn(H, D, seed=2)

    np.random.seed(44)
    X1 = np.random.randn(T, D)
    X2 = np.random.randn(T, D)
    y_true = 1

    # Persistent path: run window 1, thread state into window 2, compute gradient.
    Sxh0, Shh0, Sbh0, h0 = zero_state(H, D)
    h1, Sxh1, Shh1, Sbh1 = rnn.rtrl_forward_persistent(
        X1, Sxh0, Shh0, Sbh0, h0
    )
    h2_p, Sxh2_p, Shh2_p, Sbh2_p = rnn.rtrl_forward_persistent(
        X2, Sxh1, Shh1, Sbh1, h1
    )
    y_pred_p = rnn.output(h2_p)
    grads_p = rnn.rtrl_compute_gradients(
        y_true, y_pred_p, h2_p, Sxh2_p, Shh2_p, Sbh2_p
    )

    # Reset path: run window 2 from zero state (this is the buggy behavior).
    h2_r, Sxh2_r, Shh2_r, Sbh2_r = rnn.rtrl_forward(X2)
    y_pred_r = rnn.output(h2_r)
    grads_r = rnn.rtrl_compute_gradients(
        y_true, y_pred_r, h2_r, Sxh2_r, Shh2_r, Sbh2_r
    )

    # Hidden state on window 2 depends on the initial state, so must already differ.
    h_diff = np.max(np.abs(h2_p - h2_r))
    assert h_diff > 1e-6, (
        f"h2 should differ between persistent and reset (got max diff {h_diff:.2e})"
    )

    # Report and assert gradient differences. At least one must differ meaningfully.
    names = ['dW_xh', 'dW_hh', 'dW_hy', 'db_h', 'db_y']
    max_diffs = []
    for name, gp, gr in zip(names, grads_p, grads_r):
        d = float(np.max(np.abs(gp - gr)))
        max_diffs.append((name, d))
        print(f"  {name}: max |grad_persistent - grad_reset| = {d:.6e}")

    any_differ = any(d > 1e-6 for _, d in max_diffs)
    assert any_differ, (
        "All gradients identical between persistent and reset paths - "
        "fix is not changing behavior."
    )

    print(f"  h2 max diff = {h_diff:.6e}")
    print("test_c_gradient_differs_on_window_2: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running RTRL persistent-state unit tests...")
    print("-" * 60)
    test_a_zero_state_equivalence()
    print()
    test_b_state_threading()
    print()
    test_c_gradient_differs_on_window_2()
    print("-" * 60)
    print("All tests passed.")

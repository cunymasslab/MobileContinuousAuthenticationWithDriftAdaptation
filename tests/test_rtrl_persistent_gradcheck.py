"""
test_rtrl_persistent_gradcheck.py

Finite-difference gradient check for rtrl_forward_persistent + rtrl_compute_gradients
across a two-window threaded stream. Verifies that the analytic gradient of the
window-2 loss w.r.t. each weight matches the numerical gradient computed by
perturbing each weight and running the full two-window forward pass.

Configuration:
  H=4 hidden units, D=3 input features, 2 windows of length 5 (T=10 total).
  eps = 1e-5 for central difference.
  atol = 1e-4, rtol = 1e-4 (conservative for float64 central diff at eps=1e-5).

Run:
    python tests/test_rtrl_persistent_gradcheck.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from model import VanillaRNN


H = 4
D = 3
T_PER_WINDOW = 5
EPS = 1e-5
ATOL = 1e-4
RTOL = 1e-4


def forward_threaded(rnn, X1, X2):
    """Two-window threaded forward; returns final (h2, Sxh2, Shh2, Sbh2)."""
    Sxh0 = np.zeros((H, H, D))
    Shh0 = np.zeros((H, H, H))
    Sbh0 = np.zeros((H, H))
    h0 = np.zeros(H)
    h1, Sxh1, Shh1, Sbh1 = rnn.rtrl_forward_persistent(X1, Sxh0, Shh0, Sbh0, h0)
    h2, Sxh2, Shh2, Sbh2 = rnn.rtrl_forward_persistent(X2, Sxh1, Shh1, Sbh1, h1)
    return h2, Sxh2, Shh2, Sbh2


def loss_of(rnn, X1, X2, y_true):
    """Window-2 loss under threaded persistent forward."""
    h2, _, _, _ = forward_threaded(rnn, X1, X2)
    y_pred = rnn.output(h2)
    return rnn.compute_loss(y_true, y_pred).item()


def analytic_grads(rnn, X1, X2, y_true):
    """Analytic gradients of window-2 loss via RTRL with threaded sensitivities."""
    h2, Sxh2, Shh2, Sbh2 = forward_threaded(rnn, X1, X2)
    y_pred = rnn.output(h2)
    return rnn.rtrl_compute_gradients(y_true, y_pred, h2, Sxh2, Shh2, Sbh2)


def numerical_grad(rnn, X1, X2, y_true, param_name):
    """Central-difference numerical gradient w.r.t. getattr(rnn, param_name)."""
    W = getattr(rnn, param_name)
    grad = np.zeros_like(W)
    it = np.nditer(W, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = float(W[idx])

        W[idx] = original + EPS
        L_plus = loss_of(rnn, X1, X2, y_true)

        W[idx] = original - EPS
        L_minus = loss_of(rnn, X1, X2, y_true)

        W[idx] = original  # restore
        grad[idx] = (L_plus - L_minus) / (2.0 * EPS)
        it.iternext()
    return grad


def main():
    np.random.seed(2026)
    rnn = VanillaRNN(input_dim=D, hidden_dim=H, output_dim=1)

    np.random.seed(2027)
    X1 = np.random.randn(T_PER_WINDOW, D)
    X2 = np.random.randn(T_PER_WINDOW, D)
    y_true = 1  # arbitrary binary label

    dW_xh_a, dW_hh_a, dW_hy_a, db_h_a, db_y_a = analytic_grads(rnn, X1, X2, y_true)

    print(f"Config: H={H}, D={D}, T={T_PER_WINDOW}x2 windows, eps={EPS:.0e}")
    print(f"Tolerance: atol={ATOL:.0e}, rtol={RTOL:.0e}")
    print("-" * 72)
    print(f"  {'param':6s} {'shape':12s} {'max|ana|':>12s} {'max|ana-num|':>14s} {'rel':>10s}  result")
    print("-" * 72)

    checks = [
        ('W_xh', dW_xh_a),
        ('W_hh', dW_hh_a),
        ('W_hy', dW_hy_a),
        ('b_h', db_h_a),
        ('b_y', db_y_a),
    ]

    all_passed = True
    for param_name, analytic in checks:
        numerical = numerical_grad(rnn, X1, X2, y_true, param_name)
        max_ana = float(np.max(np.abs(analytic)))
        max_abs_diff = float(np.max(np.abs(analytic - numerical)))
        max_rel = max_abs_diff / (max_ana + 1e-12)
        passed = np.allclose(analytic, numerical, atol=ATOL, rtol=RTOL)
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        print(f"  {param_name:6s} {str(analytic.shape):12s} "
              f"{max_ana:12.3e} {max_abs_diff:14.3e} {max_rel:10.3e}  {status}")

    print("-" * 72)
    if all_passed:
        print("Gradient check: PASS (RTRL persistent analytic grad matches finite difference)")
    else:
        print("Gradient check: FAIL — analytic gradient does not match finite difference")
        sys.exit(1)


if __name__ == '__main__':
    main()

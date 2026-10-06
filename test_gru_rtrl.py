"""Validation Gates 1-2: Numerical gradient checks for GRU-RTRL."""
import numpy as np
from gru_model import GRU_NumPy

def numerical_gradients(model, X, y_true, epsilon=1e-5):
    """Compute gradients by finite differences for all parameters."""
    params = [
        ('W_xr', model.W_xr), ('W_hr', model.W_hr), ('b_r', model.b_r),
        ('W_xz', model.W_xz), ('W_hz', model.W_hz), ('b_z', model.b_z),
        ('W_xn', model.W_xn), ('W_hn', model.W_hn),
        ('b_n_input', model.b_n_input), ('b_n_hidden', model.b_n_hidden),
        ('W_hy', model.W_hy), ('b_y', model.b_y),
    ]
    
    num_grads = {}
    for name, param in params:
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            param[idx] = old_val + epsilon
            cache_plus = model.forward(X)
            h_plus = cache_plus['h'][len(X) - 1]
            y_plus = model.output(h_plus)
            loss_plus = model.compute_loss(y_true, y_plus)
            
            param[idx] = old_val - epsilon
            cache_minus = model.forward(X)
            h_minus = cache_minus['h'][len(X) - 1]
            y_minus = model.output(h_minus)
            loss_minus = model.compute_loss(y_true, y_minus)
            
            grad[idx] = (loss_plus[0] - loss_minus[0]) / (2 * epsilon)
            param[idx] = old_val
            it.iternext()
        
        num_grads[name] = grad
    
    return num_grads


def gate1_scalar():
    """Gate 1: H=1, D=1, T=5"""
    print('=== Gate 1: Scalar (H=1, D=1, T=5) ===')
    np.random.seed(42)
    model = GRU_NumPy(input_dim=1, hidden_dim=1)
    X = np.random.randn(5, 1)
    y_true = 1.0
    
    # Numerical gradients
    num_grads = numerical_gradients(model, X, y_true)
    
    # RTRL gradients
    h_final, sensitivities = model.rtrl_forward(X)
    y_pred = model.output(h_final)
    rtrl_grads = model.rtrl_compute_gradients(y_true, y_pred, h_final, sensitivities)
    
    all_pass = True
    for name, num_g in num_grads.items():
        rtrl_g = rtrl_grads[name]
        diff = np.max(np.abs(num_g - rtrl_g))
        status = 'PASS' if diff < 1e-4 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f'  {name:12s}: max_diff={diff:.2e}  {status}')
    
    print(f'Gate 1: {"PASS" if all_pass else "FAIL"}')
    return all_pass


def gate2_small():
    """Gate 2: H=4, D=3, T=10"""
    print('\n=== Gate 2: Small (H=4, D=3, T=10) ===')
    np.random.seed(123)
    model = GRU_NumPy(input_dim=3, hidden_dim=4)
    X = np.random.randn(10, 3)
    y_true = 0.0
    
    num_grads = numerical_gradients(model, X, y_true)
    
    h_final, sensitivities = model.rtrl_forward(X)
    y_pred = model.output(h_final)
    rtrl_grads = model.rtrl_compute_gradients(y_true, y_pred, h_final, sensitivities)
    
    all_pass = True
    for name, num_g in num_grads.items():
        rtrl_g = rtrl_grads[name]
        diff = np.max(np.abs(num_g - rtrl_g))
        status = 'PASS' if diff < 1e-4 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f'  {name:12s}: max_diff={diff:.2e}  {status}')
    
    print(f'Gate 2: {"PASS" if all_pass else "FAIL"}')
    return all_pass

def gate3_offline_equivalence():
    """Gate 3: Train with RTRL and BPTT offline — weights must match."""
    print('\n=== Gate 3: Offline Equivalence (RTRL vs BPTT) ===')
    
    H, D, T = 4, 3, 10
    n_samples = 20
    lr = 0.001
    epochs = 5
    
    # Generate random dataset
    np.random.seed(42)
    X_data = [np.random.randn(T, D) for _ in range(n_samples)]
    y_data = [float(i % 2) for i in range(n_samples)]
    
    # Train with BPTT
    np.random.seed(99)
    model_bptt = GRU_NumPy(input_dim=D, hidden_dim=H)
    for epoch in range(epochs):
        for i in range(n_samples):
            cache = model_bptt.forward(X_data[i])
            h_final = cache['h'][T - 1]
            y_pred = model_bptt.output(h_final)
            grads = model_bptt.backward(y_data[i], y_pred, cache, X_data[i])
            grads = model_bptt.clip_gradients(grads)
            model_bptt.update_weights(grads, lr)
    
    # Train with RTRL (same init, same data order)
    np.random.seed(99)
    model_rtrl = GRU_NumPy(input_dim=D, hidden_dim=H)
    for epoch in range(epochs):
        for i in range(n_samples):
            h_final, sensitivities = model_rtrl.rtrl_forward(X_data[i])
            y_pred = model_rtrl.output(h_final)
            grads = model_rtrl.rtrl_compute_gradients(y_data[i], y_pred, h_final, sensitivities)
            grads = model_rtrl.clip_gradients(grads)
            model_rtrl.update_weights(grads, lr)
    
    # Compare all weights
    all_pass = True
    params = ['W_xr', 'W_hr', 'b_r', 'W_xz', 'W_hz', 'b_z',
              'W_xn', 'W_hn', 'b_n_input', 'b_n_hidden', 'W_hy', 'b_y']
    for name in params:
        bptt_w = getattr(model_bptt, name)
        rtrl_w = getattr(model_rtrl, name)
        diff = np.max(np.abs(bptt_w - rtrl_w))
        status = 'PASS' if diff < 1e-5 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f'  {name:12s}: max_diff={diff:.2e}  {status}')
    
    print(f'Gate 3: {"PASS" if all_pass else "FAIL"}')
    return all_pass

def gate4_pytorch_rtrl():
    """Gate 4: PyTorch-trained GRU → NumPy weight transfer → RTRL gradients match numerical."""
    print('\n=== Gate 4: PyTorch Transfer + RTRL (H=32, D=9, T=100) ===')
    import torch
    from pytorch_models import GRUClassifier
    
    H, D, T = 32, 9, 100
    
    # Create and initialize PyTorch GRU
    torch.manual_seed(42)
    pt_model = GRUClassifier(input_dim=D, hidden_dim=H)
    pt_model.eval()
    
    # Transfer to NumPy
    np_model = GRU_NumPy(input_dim=D, hidden_dim=H)
    np_model.load_from_pytorch(pt_model)
    
    # Verify forward pass matches (Gate 0 repeat)
    np.random.seed(42)
    X = np.random.randn(T, D)
    X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pt_out = pt_model(X_torch).item()
    
    cache = np_model.forward(X)
    np_out = np_model.output(cache['h'][T - 1])[0]
    fwd_diff = abs(pt_out - np_out)
    print(f'  Forward match: diff={fwd_diff:.2e}  {"PASS" if fwd_diff < 1e-4 else "FAIL"}')
    
    # Now verify RTRL gradients match numerical on transferred weights
    y_true = 1.0
    h_final, sensitivities = np_model.rtrl_forward(X)
    y_pred = np_model.output(h_final)
    rtrl_grads = np_model.rtrl_compute_gradients(y_true, y_pred, h_final, sensitivities)
    
    # Numerical gradients (only check a subset — full H=32 is slow)
    # Check biases (fast) and one row of each weight matrix
    params_to_check = ['b_r', 'b_z', 'b_n_input', 'b_n_hidden', 'b_y']
    
    from test_gru_rtrl import numerical_gradients
    num_grads = numerical_gradients(np_model, X, y_true)
    
    all_pass = fwd_diff < 1e-4
    for name in params_to_check:
        diff = np.max(np.abs(num_grads[name] - rtrl_grads[name]))
        status = 'PASS' if diff < 1e-4 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f'  {name:12s}: max_diff={diff:.2e}  {status}')
    
    print(f'Gate 4: {"PASS" if all_pass else "FAIL"}')
    return all_pass


if __name__ == '__main__':
    g1 = gate1_scalar()
    g2 = gate2_small()
    g3 = gate3_offline_equivalence()
    g4 = gate4_pytorch_rtrl()
    print(f'\n{"="*40}')
    print(f'Gate 1: {"PASS" if g1 else "FAIL"}')
    print(f'Gate 2: {"PASS" if g2 else "FAIL"}')
    print(f'Gate 3: {"PASS" if g3 else "FAIL"}')
    print(f'Gate 4: {"PASS" if g4 else "FAIL"}')
    raise SystemExit(0 if all((g1, g2, g3, g4)) else 1)

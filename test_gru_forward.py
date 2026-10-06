"""Gate 0: Verify NumPy GRU matches PyTorch GRU output after weight transfer."""
import numpy as np
import torch
from pytorch_models import GRUClassifier
from gru_model import GRU_NumPy

def test_weight_transfer():
    H, D, T = 32, 9, 100
    seed = 42
    
    # Train or create a PyTorch GRU with known weights
    torch.manual_seed(seed)
    pt_model = GRUClassifier(input_dim=D, hidden_dim=H)
    pt_model.eval()
    
    # Create NumPy GRU and load weights
    np_model = GRU_NumPy(input_dim=D, hidden_dim=H)
    np_model.load_from_pytorch(pt_model)
    
    # Random input
    np.random.seed(seed)
    X = np.random.randn(T, D)
    X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, T, D)
    
    # PyTorch forward
    with torch.no_grad():
        pt_out = pt_model(X_torch).item()
    
    # NumPy forward
    cache = np_model.forward(X)
    h_final = cache['h'][T - 1]
    np_out = np_model.output(h_final)[0]
    
    diff = abs(pt_out - np_out)
    print(f'PyTorch output: {pt_out:.10f}')
    print(f'NumPy output:   {np_out:.10f}')
    print(f'Difference:     {diff:.2e}')
    
    if diff < 1e-4:
        print('PASS: Weight transfer validated')
    else:
        print('FAIL: Outputs do not match')
    
    return diff < 1e-4


if __name__ == '__main__':
    raise SystemExit(0 if test_weight_transfer() else 1)

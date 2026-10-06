"""
pytorch_models.py
PyTorch LSTM and GRU for Stage 3 architecture search.
Training only — frozen evaluation uses these models with no weight updates.
Mirrors the NumPy VanillaRNN training protocol exactly.
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, T, input_dim) or (T, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, hidden_dim)
        out = torch.sigmoid(self.fc(h_n.squeeze(0)))  # (batch, 1)
        return out


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, h_n = self.gru(x)  # h_n: (1, batch, hidden_dim)
        out = torch.sigmoid(self.fc(h_n.squeeze(0)))  # (batch, 1)
        return out


class VanillaRNNClassifier(nn.Module):
    """PyTorch VanillaRNN for consistent comparison. tanh activation."""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, h_n = self.rnn(x)  # h_n: (1, batch, hidden_dim)
        out = torch.sigmoid(self.fc(h_n.squeeze(0)))
        return out


def create_model(arch, input_dim, hidden_dim):
    if arch == 'vanilla_rnn':
        return VanillaRNNClassifier(input_dim, hidden_dim)
    elif arch == 'lstm':
        return LSTMClassifier(input_dim, hidden_dim)
    elif arch == 'gru':
        return GRUClassifier(input_dim, hidden_dim)
    else:
        raise ValueError(f'Unknown architecture: {arch}')


def train_pytorch_model(X_train, y_train, X_val, y_val, arch, input_dim, hidden_dim,
                        lr=0.01, max_epochs=30, patience=3, es_delta=1e-3, seed=42):
    """
    Train a PyTorch model with early stopping on validation loss.
    Matches the Stage 1 protocol exactly.

    Args:
        X_train, y_train: numpy arrays (N, T, D) and (N,)
        X_val, y_val: numpy arrays for validation
        arch: 'vanilla_rnn', 'lstm', or 'gru'
        input_dim: number of features
        hidden_dim: hidden units
        lr: learning rate (locked from Stage 1)
        max_epochs: max training epochs
        patience: early stopping patience
        es_delta: minimum improvement threshold
        seed: random seed

    Returns:
        model: trained PyTorch model (best weights restored)
        convergence_epoch: epoch where training stopped
        final_loss: best validation loss
        training_curves: list of dicts with per-epoch metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(arch, input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    training_curves = []
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    convergence_epoch = max_epochs

    max_norm = 1.0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        correct = 0

        perm = np.random.permutation(len(X_train))
        for i in perm:
            optimizer.zero_grad()
            y_pred = model(X_train_t[i])
            loss = criterion(y_pred, y_train_t[i].unsqueeze(0))
            loss.backward()
            # Gradient clipping to match NumPy protocol
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            total_loss += loss.item()
            predicted_label = 1 if y_pred.item() > 0.5 else 0
            if predicted_label == y_train[i]:
                correct += 1

        avg_train_loss = total_loss / len(X_train)
        train_accuracy = correct / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in range(len(X_val)):
                y_pred = model(X_val_t[i])
                val_loss += criterion(y_pred, y_val_t[i].unsqueeze(0)).item()
            avg_val_loss = val_loss / len(X_val)

        training_curves.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
        })

        print(f'    [{arch} H={hidden_dim}] Epoch {epoch + 1}/{max_epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {train_accuracy:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss - es_delta:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                convergence_epoch = epoch + 1
                print(f'    Early stopping at epoch {convergence_epoch}')
                break
    else:
        convergence_epoch = max_epochs

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, convergence_epoch, best_val_loss, training_curves


def evaluate_pytorch_model(model, X_test, y_test):
    """
    Run frozen inference and return prediction scores.

    Args:
        model: trained PyTorch model
        X_test, y_test: numpy arrays

    Returns:
        y_scores: numpy array of predicted probabilities
    """
    device = next(model.parameters()).device
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    model.eval()
    y_scores = []
    with torch.no_grad():
        for i in range(len(X_test)):
            y_pred = model(X_test_t[i])
            y_scores.append(y_pred.item())

    return np.array(y_scores)

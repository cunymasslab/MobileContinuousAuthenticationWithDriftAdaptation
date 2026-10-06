import numpy as np

class GRU_NumPy:
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Reset gate
        self.W_xr = np.random.randn(hidden_dim, input_dim) * np.sqrt(1.0 / (input_dim + hidden_dim))
        Q, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
        self.W_hr = Q.copy()
        self.b_r = np.zeros(hidden_dim)
        
        # Update gate
        self.W_xz = np.random.randn(hidden_dim, input_dim) * np.sqrt(1.0 / (input_dim + hidden_dim))
        Q, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
        self.W_hz = Q.copy()
        self.b_z = np.zeros(hidden_dim)
        
        # Candidate — TWO biases to match PyTorch
        self.W_xn = np.random.randn(hidden_dim, input_dim) * np.sqrt(1.0 / (input_dim + hidden_dim))
        Q, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
        self.W_hn = Q.copy()
        self.b_n_input = np.zeros(hidden_dim)   # added to input term
        self.b_n_hidden = np.zeros(hidden_dim)  # multiplied by r_t with recurrent term
        
        # Output layer
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(1.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)
    
    def sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    def forward_step(self, x_t, h_prev):
        """Single timestep. Returns r_t, z_t, n_t, h_t."""
        # Reset gate
        r_t = self.sigmoid(self.W_xr @ x_t + self.W_hr @ h_prev + self.b_r)
        
        # Update gate
        z_t = self.sigmoid(self.W_xz @ x_t + self.W_hz @ h_prev + self.b_z)
        
        # Candidate — NOTE: b_n_hidden is INSIDE the reset gate
        n_t = np.tanh(self.W_xn @ x_t + self.b_n_input + r_t * (self.W_hn @ h_prev + self.b_n_hidden))
        
        # New hidden state
        h_t = (1 - z_t) * n_t + z_t * h_prev
        
        return r_t, z_t, n_t, h_t
    
    def forward(self, X_sequence):
        """Full sequence forward pass. Returns caches for BPTT."""
        T = len(X_sequence)
        h_t = np.zeros(self.hidden_dim)
        
        cache = {'r': {}, 'z': {}, 'n': {}, 'h': {}}
        cache['h'][-1] = h_t.copy()  # h_{-1} = zeros
        
        for t in range(T):
            r_t, z_t, n_t, h_t = self.forward_step(X_sequence[t], h_t)
            cache['r'][t] = r_t
            cache['z'][t] = z_t
            cache['n'][t] = n_t
            cache['h'][t] = h_t
        
        return cache
    
    def output(self, h_final):
        return self.sigmoid(self.W_hy @ h_final + self.b_y)
    
    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def load_from_pytorch(self, pytorch_model):
        """Load weights from a trained PyTorch GRUClassifier."""
        sd = pytorch_model.state_dict()
        
        W_ih = sd['gru.weight_ih_l0'].detach().cpu().numpy().astype(np.float64)
        W_hh = sd['gru.weight_hh_l0'].detach().cpu().numpy().astype(np.float64)
        b_ih = sd['gru.bias_ih_l0'].detach().cpu().numpy().astype(np.float64)
        b_hh = sd['gru.bias_hh_l0'].detach().cpu().numpy().astype(np.float64)
        
        H = self.hidden_dim
        # PyTorch order: reset, update, new
        self.W_xr = W_ih[:H]
        self.W_xz = W_ih[H:2*H]
        self.W_xn = W_ih[2*H:]
        
        self.W_hr = W_hh[:H]
        self.W_hz = W_hh[H:2*H]
        self.W_hn = W_hh[2*H:]
        
        # Reset and update biases: safe to sum
        self.b_r = b_ih[:H] + b_hh[:H]
        self.b_z = b_ih[H:2*H] + b_hh[H:2*H]
        
        # Candidate biases: DO NOT sum — keep separate
        self.b_n_input = b_ih[2*H:]
        self.b_n_hidden = b_hh[2*H:]
        
        # Output layer
        self.W_hy = sd['fc.weight'].detach().cpu().numpy().astype(np.float64)
        self.b_y = sd['fc.bias'].detach().cpu().numpy().astype(np.float64)

    def rtrl_forward(self, X_sequence):
        T = len(X_sequence)
        H = self.hidden_dim
        D = self.input_dim
        
        # Sensitivity tensors: dh/dtheta for each parameter
        S = {
            'W_xr': np.zeros((H, H, D)), 'W_hr': np.zeros((H, H, H)), 'b_r': np.zeros((H, H)),
            'W_xz': np.zeros((H, H, D)), 'W_hz': np.zeros((H, H, H)), 'b_z': np.zeros((H, H)),
            'W_xn': np.zeros((H, H, D)), 'W_hn': np.zeros((H, H, H)),
            'b_n_input': np.zeros((H, H)), 'b_n_hidden': np.zeros((H, H)),
        }
        
        h_t = np.zeros(H)
        
        for t in range(T):
            h_prev = h_t
            x_t = X_sequence[t]
            r_t, z_t, n_t, h_t = self.forward_step(x_t, h_prev)
            
            # Gate derivatives
            dz = z_t * (1 - z_t)    # (H,)
            dr = r_t * (1 - r_t)    # (H,)
            dn = 1 - n_t ** 2       # (H,)
            
            S_new = {}
            
            for key in S:
                S_prev = S[key]
                is_3d = S_prev.ndim == 3
                
                # Common recurrent terms: W_hz @ S_prev, W_hr @ S_prev
                if is_3d:
                    hz_S = np.einsum('im,mjk->ijk', self.W_hz, S_prev)
                    hr_S = np.einsum('im,mjk->ijk', self.W_hr, S_prev)
                else:
                    hz_S = self.W_hz @ S_prev
                    hr_S = self.W_hr @ S_prev
                
                # da_z = direct_z + W_hz @ S_prev
                da_z = hz_S
                if key == 'W_xz':
                    direct = np.zeros_like(S_prev)
                    for i in range(H): direct[i, i, :] = x_t
                    da_z = da_z + direct
                elif key == 'W_hz':
                    direct = np.zeros_like(S_prev)
                    for i in range(H): direct[i, i, :] = h_prev
                    da_z = da_z + direct
                elif key == 'b_z':
                    da_z = da_z + np.eye(H)
                
                # da_r = direct_r + W_hr @ S_prev
                da_r = hr_S
                if key == 'W_xr':
                    direct = np.zeros_like(S_prev)
                    for i in range(H): direct[i, i, :] = x_t
                    da_r = da_r + direct
                elif key == 'W_hr':
                    direct = np.zeros_like(S_prev)
                    for i in range(H): direct[i, i, :] = h_prev
                    da_r = da_r + direct
                elif key == 'b_r':
                    da_r = da_r + np.eye(H)
                
                # dz/dtheta and dr/dtheta
                if is_3d:
                    dz_dt = dz.reshape(H, 1, 1) * da_z
                    dr_dt = dr.reshape(H, 1, 1) * da_r
                else:
                    dz_dt = dz.reshape(H, 1) * da_z
                    dr_dt = dr.reshape(H, 1) * da_r
                
                # da_n: derivative of candidate pre-activation
                # a_n[i] = ... + r_t[i] * q_t[i], where q_t = W_hn @ h_prev + b_n_hidden
                # Product rule: d(r*q)/dθ = dr/dθ * q + r * dq/dθ
                q_t = self.W_hn @ h_prev + self.b_n_hidden  # (H,)
                
                # dq/dθ = W_hn @ S_prev + [direct for W_hn or b_n_hidden]
                if is_3d:
                    dq = np.einsum('im,mjk->ijk', self.W_hn, S_prev)
                else:
                    dq = self.W_hn @ S_prev
                
                if key == 'W_hn':
                    direct = np.zeros_like(S_prev)
                    for i in range(H): direct[i, i, :] = h_prev
                    dq = dq + direct
                elif key == 'b_n_hidden':
                    dq = dq + np.eye(H)
                
                # d(r*q)/dθ = dr/dθ * q + r * dq/dθ
                if is_3d:
                    da_n = dr_dt * q_t.reshape(H, 1, 1) + r_t.reshape(H, 1, 1) * dq
                else:
                    da_n = dr_dt * q_t.reshape(H, 1) + r_t.reshape(H, 1) * dq
                
                # Add direct terms outside the r*q product
                if key == 'W_xn':
                    direct = np.zeros_like(S_prev)
                    for i in range(H): direct[i, i, :] = x_t
                    da_n = da_n + direct
                elif key == 'b_n_input':
                    da_n = da_n + np.eye(H)

                
                # dn/dtheta
                if is_3d:
                    dn_dt = dn.reshape(H, 1, 1) * da_n
                else:
                    dn_dt = dn.reshape(H, 1) * da_n
                
                # Final: S_new = dz/dt * (h_prev - n_t) + (1-z_t) * dn/dt + z_t * S_prev
                if is_3d:
                    S_new[key] = (dz_dt * (h_prev - n_t).reshape(H, 1, 1)
                                + (1 - z_t).reshape(H, 1, 1) * dn_dt
                                + z_t.reshape(H, 1, 1) * S_prev)
                else:
                    S_new[key] = (dz_dt * (h_prev - n_t).reshape(H, 1)
                                + (1 - z_t).reshape(H, 1) * dn_dt
                                + z_t.reshape(H, 1) * S_prev)
            
            S = S_new
        
        return h_t, S
    
    def clip_gradients(self, grads, max_norm=1.0):
        total = sum(np.sum(g**2) for g in grads.values())
        global_norm = np.sqrt(total)
        if global_norm > max_norm:
            scale = max_norm / global_norm
            grads = {k: v * scale for k, v in grads.items()}
        return grads

    def update_weights(self, grads, learning_rate):
        for key in grads:
            param = getattr(self, key)
            param -= learning_rate * grads[key]

    def rtrl_compute_gradients(self, y_true, y_pred, h_final, sensitivities):
        # Output layer gradients (same as VanillaRNN)
        dL_dz = y_pred - y_true
        dW_hy = np.outer(dL_dz, h_final)
        db_y = dL_dz
        
        # Gradient into final hidden state
        dL_dh = self.W_hy.T @ dL_dz   # (H,)
        
        # Use sensitivity matrices to get recurrent gradients
        grads = {}
        for key, S in sensitivities.items():
            if S.ndim == 3:
                grads[key] = np.tensordot(dL_dh, S, axes=([0], [0]))
            else:
                grads[key] = dL_dh @ S
        
        grads['W_hy'] = dW_hy
        grads['b_y'] = db_y
        
        return grads

    def backward(self, y_true, y_pred, cache, X_sequence):
        """BPTT for GRU. Returns gradients dict matching rtrl_compute_gradients format."""
        T = len(X_sequence)
        H = self.hidden_dim

        # Initialize gradient accumulators
        grads = {k: np.zeros_like(getattr(self, k)) for k in [
            'W_xr', 'W_hr', 'b_r', 'W_xz', 'W_hz', 'b_z',
            'W_xn', 'W_hn', 'b_n_input', 'b_n_hidden', 'W_hy', 'b_y'
        ]}

        # Output layer gradients
        dL_dz = y_pred - y_true
        grads['W_hy'] = np.outer(dL_dz, cache['h'][T - 1])
        grads['b_y'] = dL_dz

        # Gradient into final hidden state
        dL_dh = self.W_hy.T @ dL_dz  # (H,)

        # Loop backward through time
        for t in range(T - 1, -1, -1):
            h_prev = cache['h'][t - 1]  # works because cache['h'][-1] = zeros
            r_t = cache['r'][t]
            z_t = cache['z'][t]
            n_t = cache['n'][t]

            # dL/dh_t is the incoming gradient
            # h_t = (1 - z_t) * n_t + z_t * h_prev
            
            # Gradient through to n_t
            dL_dn = dL_dh * (1 - z_t)

            # Gradient through to z_t
            dL_dz_gate = dL_dh * (h_prev - n_t)

            # ---- Candidate gate: n_t = tanh(a_n) ----
            # a_n = W_xn @ x_t + b_n_input + r_t * (W_hn @ h_prev + b_n_hidden)
            dn = 1 - n_t ** 2  # tanh derivative
            dL_da_n = dL_dn * dn

            grads['W_xn'] += np.outer(dL_da_n, X_sequence[t])
            grads['b_n_input'] += dL_da_n

            # The recurrent part: r_t * (W_hn @ h_prev + b_n_hidden)
            q_t = self.W_hn @ h_prev + self.b_n_hidden
            dL_d_rq = dL_da_n  # gradient into r_t * q_t

            # d(r*q)/dr = q, d(r*q)/dq = r
            dL_dr_from_n = dL_d_rq * q_t
            dL_dq = dL_d_rq * r_t

            grads['W_hn'] += np.outer(dL_dq, h_prev)
            grads['b_n_hidden'] += dL_dq

            # ---- Update gate: z_t = sigmoid(a_z) ----
            dz = z_t * (1 - z_t)
            dL_da_z = dL_dz_gate * dz

            grads['W_xz'] += np.outer(dL_da_z, X_sequence[t])
            grads['W_hz'] += np.outer(dL_da_z, h_prev)
            grads['b_z'] += dL_da_z

            # ---- Reset gate: r_t = sigmoid(a_r) ----
            dr = r_t * (1 - r_t)
            dL_da_r = dL_dr_from_n * dr

            grads['W_xr'] += np.outer(dL_da_r, X_sequence[t])
            grads['W_hr'] += np.outer(dL_da_r, h_prev)
            grads['b_r'] += dL_da_r

            # ---- Propagate gradient to h_{t-1} ----
            # h_t depends on h_prev through: z_t * h_prev, and through z_t, r_t, n_t
            dL_dh_prev = dL_dh * z_t                          # direct path: z * h_prev
            dL_dh_prev += self.W_hz.T @ dL_da_z               # through z gate
            dL_dh_prev += self.W_hr.T @ dL_da_r               # through r gate
            dL_dh_prev += self.W_hn.T @ dL_dq                 # through candidate (via q)

            dL_dh = dL_dh_prev

        return grads



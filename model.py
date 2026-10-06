import numpy as np

class VanillaRNN:
    """
    Vanilla RNN for binary classification of sensor data windows.
    Uses BPTT for training. Built from scratch with numpy. 
    """
    def __init__(self, input_dim, hidden_dim, output_dim = 1):
        """
        Initializes weight matrices and biases.

        Args:
            input_dim: int, number of input features per timestep
            hidden_dim: int, number of hidden units
            output_dim: int, number of outputs (default 1 for binary classification)

        Weight initialization:
            W_xh, W_hy: Xavier initialization
            W_hh: Orthogonal initialization (preservers gradient norms)
            b_h, b_y: Zeros
        """
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(1.0 / (input_dim + hidden_dim))
        
        # Using orthogonal initialization for weights in the hidden layer
        random_matrix = np.random.randn(hidden_dim, hidden_dim) 
        Q, R = np.linalg.qr(random_matrix)
        self.W_hh = Q
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(1.0 / (hidden_dim + output_dim))

        self.b_h =  np.zeros(hidden_dim,)
        self.b_y = np.zeros(output_dim,)

    def forward_step(self, x_t, h_prev):
        """
        Single timestep forward pass.

        Args:
            x_t: input vector at time t, shape (input_dim, )
            h_prev: previous hidden state, shape (hidden_dim, )

        Returns:
            a_t: pre-activation, shape (hidden_dim)
            h_t: hidden state after tanh, shape (hidden_dim, )
        """
        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        h_t = np.tanh(a_t)
        return a_t, h_t

    def forward(self, x_t):
        """
        Processes a full sequence through the RNN, storing states for BPTT

        Args:
            X_sequence: numpy array of shape (T, input_dim), one window of sensor data

        Returns:
            a_cache: dict mapping timestep -> pre-activation vector
            h_cache: dict mapping timestep -> hidden state vector    
        """
        # T -> number of time steps in the sequence
        T = len(x_t)
        h_t = np.zeros_like(self.b_h)

        a_cache = {}
        h_cache = {}

        for t in range(T):
            a_t, h_t = self.forward_step(x_t[t], h_t)
            a_cache[t] = a_t
            h_cache[t] = h_t 

        return a_cache, h_cache 
    
    def sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0,1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


    def output(self, h_final):
        """
        Computes prediction from final hidden state.

        Args:
            h_final: last hidden state, shape (hidden_dim,)
        
        Returns:
            y_pred: prediction between 0 and 1, shape (output_dim, )
        """
        y_pred = self.sigmoid(self.W_hy @ h_final + self.b_y)
        return y_pred 

    def compute_loss(self, y_true, y_pred):
        # clip to avoid log 0
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_true, y_pred, h_cache, a_cache, X_sequence):
        """
        Backpropagation Through Time (BPTT). Computes gradients for all weights
        by propagating the error from the output backward through all timesteps.

        Args:
            y_true: true label (0 or 1)
            y_pred: predicted probability, shape (output_dim, )
            h_cache: dict mapping timestep -> hidden state vector 
            a_cache: dict mapping timestep -> pre-activation vector
            X_sequence: numpy array of shape (T, input_dim), the input window

        Returns:
            tuple of (dW_xh, dW_hh, dW_hy, db_h, db_y), gradients for each parameter
        """
        T = len(X_sequence)

        # Initialize gradient accumulators to zero
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # Step 1: Output layer gradients
        dL_dz = y_pred - y_true    # shape (output_dim,)
        dW_hy = np.outer(dL_dz, h_cache[T-1])  # outer product
        db_y = dL_dz 

        # Step 2: Gradient into final hidden state 
        dL_dh = self.W_hy.T @ dL_dz     # shape (hidden_dim, )

        # Step 3: Loop backward through time 
        for t in range(T -1, -1, -1):
            # Through tanh
            dL_da = dL_dh * (1 - h_cache[t]**2) # element-wise 

            # Accumulate weight gradients 
            dW_xh += np.outer(dL_da, X_sequence[t]) # outer product 

            if t > 0:
                dW_hh += np.outer(dL_da, h_cache[t-1])# outer product 

            else:
                dW_hh += np.outer(dL_da, np.zeros_like(h_cache[t]))

            db_h += dL_da

            # Propagate gradient to previous timestep 
            dL_dh = self.W_hh.T @ dL_da

        return dW_xh, dW_hh, dW_hy, db_h, db_y 

    def clip_gradients(self, grads, max_norm = 1.0):
        """
        Clips gradients by global norm to prevent exploding gradients.

        If the total norm exceeds max_norm, all gradients are scaled down proportionally.

        Args:
            grads: tuple of (dW_xh, dW_hh, dW_hy, db_h, db_y)
            max_norm: float, maximum allowed global norm (default 1.0)

        Returns: 
            tuple of clipped gradients, same format as input
        
        """
        dW_xh, dW_hh, dW_hy, db_h, db_y = grads 

        total = np.sum(dW_xh**2) + np.sum(dW_hh**2) + np.sum(dW_hy**2) + np.sum(db_h**2) + np.sum(db_y**2)
        global_norm = np.sqrt(total)

        if global_norm > max_norm:
            scale = max_norm / global_norm
            dW_xh *= scale 
            dW_hh *= scale 
            dW_hy *= scale 
            db_h *= scale 
            db_y *= scale 

        return dW_xh, dW_hh, dW_hy, db_h, db_y

    def update_weights(self, grads, learning_rate):
        """
        SGD weight update. Subtracts learning_rate * gradient from each parameter 

        Args: 
            grads: tuple of (dW_xh, dW_hh, dW_hy, db_h, db_y)
            learning_rate: float, step size for gradient descent
        """
        dW_xh, dW_hh, dW_hy, db_h, db_y = grads 

        self.W_xh -= learning_rate * dW_xh 
        self.W_hy -= learning_rate * dW_hy 
        self.W_hh -= learning_rate * dW_hh
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

    def rtrl_forward(self, X_sequence):
        '''
        Forward pass with RTRL sensitivity matrix tracking. 
        Computes hidden states while maintaining sensitivity matrices
        that track how each weight affects each hidden unit over time.

        Args: 
            X_sequence: numpy array of shape (T, input_dim), one window of sensor data

        Returns:
            h_t: final hidden state, shape (hidden_dim, )
            S_xh: sensitivity matrix for W_xh, shape (H, H, D)
            S_hh: sensitivity matrix for W_hh, shape (H, H, H)
            S_bh: sensitivity matrix for b_h, shape (H, H)
        
        '''
        T = len(X_sequence)
        H = len(self.b_h)
        D = X_sequence.shape[1]

        S_xh = np.zeros((H, H, D))
        S_hh = np.zeros((H, H, H))
        S_bh = np.zeros((H, H))

        # hidden state
        h_t = np.zeros_like(self.b_h)

        direct_xh = np.zeros((H, H, D))
        direct_hh = np.zeros((H, H, H))

        for t in range(T):
            h_prev = h_t
            a_t, h_t = self.forward_step(X_sequence[t], h_prev)
            d_t = 1 - h_t ** 2 # tanh derivative, shape (H, )

            
            for i in range(H):
                # computing the direct effect for x_h
                direct_xh[i, i, :] = X_sequence[t]

                # computing the direct effect for hh
                direct_hh[i, i, :] = h_prev

            # direct effect for b_h
            direct_bh = np.eye(H)

            # Update sensitivity matrices
            S_xh = d_t.reshape(H, 1, 1) * (direct_xh + np.einsum('im, mjk->ijk', self.W_hh, S_xh))
            S_hh = d_t.reshape(H, 1, 1) * (direct_hh + np.einsum('im,mjk->ijk', self.W_hh, S_hh))
            S_bh = d_t.reshape(H, 1) * (direct_bh + self.W_hh @ S_bh)
            
        return h_t, S_xh, S_hh, S_bh 


    def rtrl_forward_persistent(self, X_sequence, S_xh_prev, S_hh_prev, S_bh_prev, h_prev):
        '''
        Forward pass with RTRL sensitivity tracking, initialized from previous state
        rather than zeros. Enables RTRL horizon to span multiple windows.

        Args:
            X_sequence: numpy array of shape (T, input_dim), one window of sensor data
            S_xh_prev: sensitivity matrix for W_xh from previous window, shape (H, H, D)
            S_hh_prev: sensitivity matrix for W_hh from previous window, shape (H, H, H)
            S_bh_prev: sensitivity matrix for b_h from previous window, shape (H, H)
            h_prev: final hidden state from previous window, shape (hidden_dim,)

        Returns:
            h_t: final hidden state, shape (hidden_dim,)
            S_xh: sensitivity matrix for W_xh, shape (H, H, D)
            S_hh: sensitivity matrix for W_hh, shape (H, H, H)
            S_bh: sensitivity matrix for b_h, shape (H, H)
        '''
        T = len(X_sequence)
        H = len(self.b_h)
        D = X_sequence.shape[1]

        S_xh = S_xh_prev.copy()
        S_hh = S_hh_prev.copy()
        S_bh = S_bh_prev.copy()

        # hidden state
        h_t = h_prev.copy()

        direct_xh = np.zeros((H, H, D))
        direct_hh = np.zeros((H, H, H))

        for t in range(T):
            h_prev_t = h_t
            a_t, h_t = self.forward_step(X_sequence[t], h_prev_t)
            d_t = 1 - h_t ** 2  # tanh derivative, shape (H,)

            for i in range(H):
                # computing the direct effect for x_h
                direct_xh[i, i, :] = X_sequence[t]

                # computing the direct effect for hh
                direct_hh[i, i, :] = h_prev_t

            # direct effect for b_h
            direct_bh = np.eye(H)

            # Update sensitivity matrices
            S_xh = d_t.reshape(H, 1, 1) * (direct_xh + np.einsum('im, mjk->ijk', self.W_hh, S_xh))
            S_hh = d_t.reshape(H, 1, 1) * (direct_hh + np.einsum('im,mjk->ijk', self.W_hh, S_hh))
            S_bh = d_t.reshape(H, 1) * (direct_bh + self.W_hh @ S_bh)

        return h_t, S_xh, S_hh, S_bh






    def rtrl_compute_gradients(self, y_true, y_pred, h_final, S_xh, S_hh, S_bh):
        '''
        Computes gradients using RTRL sensitivity matrices.
        Produces identical gradients to BPTT but computed via forward accumulation.

        Args:
            y_true: true label (0 or 1)
            y_pred: predicted probability, shape (output_dim,)
            h_final: final hidden state, shape (hidden_dim, )
            S_xh: sensitivity matrix for W_xh from rtrl forward
            S_hh: sensitivity matrix for W_hh from rtrl forward
            S_bh: sensitivity matrix for b_h from rtrl forward

        Returns:
            tuple of(dW_xh, dW_hh, dW_hy, db_h, db_y), gradients for each parameter        
        '''

        # computing output layer gradients
        dL_dZ = y_pred - y_true

        # output layer gradients (same as bptt)
        dW_hy = np.outer(dL_dZ, h_final)
        db_y = dL_dZ 

        # gradient into final hidden state
        dL_dh = self.W_hy.T @ dL_dZ

        # use sensitivity matrices to get recurrent gradients
        # using tensordot to sum over the first axis of both arrays
        # dL_dh is (H,), S_xh is (H, H, D) - contracting over H gives (H, D) which matches W_xh's shape
        dW_xh = np.tensordot(dL_dh, S_xh, axes = ([0], [0]))
        dW_hh = np.tensordot(dL_dh, S_hh, axes = ([0], [0]))
        db_h = dL_dh @ S_bh 

        return dW_xh, dW_hh, dW_hy, db_h, db_y 


# Driver / Test code
# from preprocessor import compute_magnitude, segment_into_windows
# from data_loader import load_session_sensors

# data = load_session_sensors('100669', 1, ['Accelerometer', 'Gyroscope', 'Magnetometer'])
# data = compute_magnitude(data)
# windows = segment_into_windows(data)

# BPTT gradients
# rnn = VanillaRNN(input_dim=12, hidden_dim=32)
# a_cache, h_cache = rnn.forward(windows[0])
# y_pred = rnn.output(h_cache[199])
# bptt_grads = rnn.backward(1, y_pred, h_cache, a_cache, windows[0])


# RTRL gradients (same weights, same input)
# h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward(windows[0])
# y_pred_rtrl = rnn.output(h_final)
# rtrl_grads = rnn.rtrl_compute_gradients(1, y_pred_rtrl, h_final, S_xh, S_hh, S_bh)

# # Compare
# for name, bptt_g, rtrl_g in zip(['dW_xh', 'dW_hh', 'dW_hy', 'db_h', 'db_y'], bptt_grads, rtrl_grads):
#     diff = np.max(np.abs(bptt_g - rtrl_g))
#     print(f'{name}: max difference = {diff:.10f}')


# loss = rnn.compute_loss(1, y_pred)

# print(f'Prediction: {y_pred}')
# print(f'Loss: {loss}')

# grads = rnn.backward(1, y_pred, h_cache, a_cache, windows[0])
# dW_xh, dW_hh, dW_hy, db_h, db_y  = grads 

# print(f'dW_xh shape: {dW_xh.shape}, max: {np.max(np.abs(dW_xh)):.6f}')
# print(f'dW_hh shape: {dW_hh.shape}, max: {np.max(np.abs(dW_hh)):.6f}')
# print(f'dW_hy shape: {dW_hy.shape}, max: {np.max(np.abs(dW_hy)):.6f}')

# print('\n\n\n')

# # Test one training step
# print('\n--- Training step test ---')
# for step in range(5):
#     a_cache, h_cache = rnn.forward(windows[0])
#     y_pred = rnn.output(h_cache[199])
#     loss = rnn.compute_loss(1, y_pred)
#     grads = rnn.backward(1, y_pred, h_cache, a_cache, windows[0])
#     grads = rnn.clip_gradients(grads)
#     rnn.update_weights(grads, learning_rate= 0.001)
#     print(f'Step: {step}: Loss = {loss[0]:.6f}, Prediction = {y_pred[0]:.6f}')


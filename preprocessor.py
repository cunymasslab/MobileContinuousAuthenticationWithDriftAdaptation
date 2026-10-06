"""
preprocessor.py
Transforms raw sensor arrays into model-ready inputs.
Handles windowing, normalization, and dataset construction.
"""

import numpy as np
from data_loader import load_session_sensors

def segment_into_windows(data, window_size=100, step_size=100):
    """
    Chops a continuous sensor array into fixed-length windows.
    
    Args:
        data: numpy array of shape (T, num_features)
        window_size: int, number of timesteps per window (default 200 ≈ 2 sec at 98 Hz)
        step_size: int, how far to advance between windows (default 200 = no overlap)
    
    Returns:
        numpy array of shape (num_windows, window_size, num_features)
    """
    sequences = []

    n = len(data)

    for i in range(0, n - window_size + 1, step_size):
        window = data[i : i + window_size]

        sequences.append(window)

    if len(sequences) == 0:
        return np.zeros((0, window_size, data.shape[1]))

    return np.array(sequences)

def fit_normalizer(windows):
    """
    Computes per-channel mean and std from training windows.
    These stats will be used to normalize both training and test data. 

    Args:
        windows: numpy array of shape(num_windows, window_size, num_features)

    Returns:
        mean: array of shape (num_features, ) - one mean per channel 
        std: array of shape (num_features, ) - one std per channel 
    """
    # Reshape from (num_windows, window_size, num_features) to (all_samples, num_features)
    # so we compute stats across all timesteps from all windows 
    windows = windows.reshape(-1, windows.shape[2])

    mean = np.mean(windows, axis = 0)
    std = np.std(windows, axis = 0)

    return mean, std

def compute_magnitude(data):
    """
    Adds a magnitude column after each group of 3 (x, y, z) sensor columns.
    Magnitude = sqrt(x² + y² + z²)
    
    Args:
        data: numpy array of shape (T, 3 * num_sensors)
    
    Returns:
        numpy array of shape (T, 4 * num_sensors) with magnitude columns inserted
    """
    result = []

    for i in range(0, data.shape[1], 3):
        xyz = data[:, i:i+3]
        mag = np.sqrt(np.sum(xyz**2, axis=1, keepdims=True))
        result.append(xyz)
        result.append(mag)

    return np.hstack(result)

def get_feature_names(sensors):
    """
    Returns ordered list of feature names matching the column layout after compute_magnitude.
    
    Args:
        sensors: list of sensor names, e.g. ['Accelerometer', 'Gyroscope']
    
    Returns:
        list of strings, e.g. ['Accelerometer_x', 'Accelerometer_y', ..., 'Gyroscope_mag']
    """
    names = []

    for sensor in sensors:
        for axis in ['x', 'y', 'z']:
            names.append(f'{sensor}_{axis}')
        names.append(f'{sensor}_mag')

    return names 

def normalize(windows, mean, std):
    """
    Applies z-score normalization using pre-computed mean and std.
    
    Args:
        windows: numpy array of shape (num_windows, window_size, num_features)
        mean: array of shape (num_features,) from fit_normalizer
        std: array of shape (num_features,) from fit_normalizer
    
    Returns:
        normalized windows, same shape as input
    """

    windows = (windows - mean) / (std  + 1e-8)
    return windows 

def create_dataset(legitmate_windows, impostor_windows, rng=None):

    """
    Combines legitimate and impostor windows into a single shuffled dataset. 

    Args:
        legitimate_windows: numpy array of shape (num_legit, window_size, num_features)
        impostor_windows: numpy array of shape (num_impostor, window_size, num_features)

    Returns:
        X: numpy array of shape (total_windows, window_size, num_features)
        y: numpy array of shape (total_windows,): 1 for legitimate, 0 for impostor
    
    """
    legitmate_labels = np.ones(len(legitmate_windows))
    impostor_labels = np.zeros(len(impostor_windows))

    X = np.concatenate([legitmate_windows, impostor_windows])
    y = np.concatenate([legitmate_labels, impostor_labels])

    if rng is None:
        rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == '__main__':

    # Testing

    # Legitimate User 
    data_legit = load_session_sensors('100669', 1, ['Accelerometer' , 'Gyroscope', 'Magnetometer'])
    data_legit = compute_magnitude(data_legit)
    legit_windows = segment_into_windows(data_legit)

    # Impostor 
    data_impostor = load_session_sensors('151985', 1, ['Accelerometer', 'Gyroscope', 'Magnetometer'])
    data_impostor = compute_magnitude(data_impostor)
    impostor_windows = segment_into_windows(data_impostor)

    # DRIVER
    # X, y = create_dataset(legit_windows, impostor_windows)
    # print(f'X shape: {X.shape}')
    # print(f'Y shape: {y.shape}')
    # print(f'Legitmate count: {int(np.sum(y))}')
    # print(f'Impostor count: {int(len(y) - np.sum(y))}')








#data = load_session_sensors('100669', 1, ['Accelerometer', 'Gyroscope', 'Magnetometer'])

# windows = segment_into_windows(data)
# mean, std = fit_normalizer(windows)
# print(f"Mean shape: {mean.shape}, values: {mean}")
# print(f"Std shape: {std.shape}, values: {std}")

# windows_normalized = normalize(windows, mean, std)

# norm_mean, norm_std = fit_normalizer(windows_normalized)

# print()
# print(f'After normalization - Mean: {norm_mean}')
# print(f'After normalization - Std: {norm_std}')

# print(f'Before magnitude: {data.shape}')
# data_with_mag = compute_magnitude(data)
# print(f'After magnitude: {data_with_mag.shape}')
# print(f'Features: {get_feature_names(['Accelerometer', 'Gyroscope', 'Magnetometer'])}')
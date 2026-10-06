"""
Experiment Runner. Loads data, trains the vanilla RNN, evaluates performance


"""

import numpy as np
from data_loader import load_session_sensors, get_sessions_by_condition
from preprocessor import compute_magnitude, segment_into_windows, fit_normalizer, normalize, create_dataset
from model import VanillaRNN
import math

# Setting Hyperparameters
TARGET_USER = '100669'
IMPOSTOR_USERS = ['151985', '171538', '180679']
ACTIVITY = 'READING'
POSTURE = 'SITTING'
SENSORS = ['Accelerometer', 'Gyroscope', 'Magnetometer']
WINDOW_SIZE = 200
STEP_SIZE = 200 
HIDDEN_DIM = 32
LEARNING_RATE = 0.001 
EPOCHS = 20
MAX_NORM = 1.0 

# Confusion Matrix
def confusion_matrix(y_true, y_pred_probs, threshold = 0.5):
    y_pred_labels = (y_pred_probs > threshold).astype(int)

    TP = np.sum((y_pred_labels == 1) & (y_true == 1))
    TN = np.sum((y_pred_labels == 0) & (y_true == 0))
    FP = np.sum((y_pred_labels == 1) & (y_true == 0))
    FN = np.sum((y_pred_labels == 0) & (y_true == 1))

    return TP, TN, FP, FN

# Computing the EER
def compute_eer(y_true, y_pred_probs):
    far_list = []
    frr_list = []

    for t in np.linspace(0, 1, 1001):
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred_probs, t)
        FAR = FP / (FP + TN) if (FP + TN) > 0 else 0 
        FRR  = FN / (FN + TP) if (FN + TP) > 0 else 0 
        far_list.append(FAR)
        frr_list.append(FRR)
  

    min_diff = float('inf')
    eer = 0

    for i in range(len(far_list)):
        diff = abs(far_list[i] - frr_list[i])

        if diff < min_diff:
            min_diff = diff 
            eer = (far_list[i] + frr_list[i]) / 2
    return eer 

# Calculating accuracy, precision, recall, f1, mcc
def evaluate(y_true, y_pred_probs, threshold = 0.5):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred_probs, threshold)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) >0 else 0 
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / denom if denom > 0 else 0 

    return accuracy, precision, recall, f1, mcc

# Load and window legitimate training data 
def load_user_windows(user_id, sessions, sensors, window_size, step_size):
    '''
    Loads sensor data for multiple sessiosn, computes magnitude, and windows everything

    Args:
        user_id: string, e.g. '100669'
        sessions: list of session numbers to load
        sensors: list of sensor names, e.g. ['Accelerometer', 'Gyroscope', 'Magnetometer']
        window_size: int, timesteps per window 
        step_size: int, stride between windows
    
    Returns:
        numpy array of shape (total_windows, window_size, 4 * num_sensors)
    '''
    # list to store windows
    all_windows = []

    for session in sessions:
        data = load_session_sensors(user_id, session, sensors)
        data = compute_magnitude(data)
        windows = segment_into_windows(data, window_size, step_size)
        all_windows.append(windows)

    return np.concatenate(all_windows)


# loading the data
session_numbers = get_sessions_by_condition(TARGET_USER, ACTIVITY, POSTURE)

# separating training and test sessions

# using the first 3 sessions as training data
train_sessions = session_numbers[:3]

# using the last  session as test data 
test_sessions = session_numbers[3]


# getting the training data for the legit user
legit_train_windows = load_user_windows(TARGET_USER, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)

# getting the training data for the impostor
impostor_train_windows = []

for user in IMPOSTOR_USERS:
    sessions = get_sessions_by_condition(user, ACTIVITY, POSTURE)[:3]
    impostor_train_windows.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))

impostor_train_windows = np.concatenate(impostor_train_windows)

# Checking the shape of the training data 
print(f'Legit Train Windows Shape: {legit_train_windows.shape}')
print(f'Impostor Train Windows Shape: {impostor_train_windows.shape}')

# Loading the Test Data 

# Getting the test data for the legit users
legit_test_windows = load_user_windows(TARGET_USER, [test_sessions], SENSORS, WINDOW_SIZE, STEP_SIZE)

# Getting the test data for the impostor users
impostor_test_windows = []

for user in IMPOSTOR_USERS:
    sessions = get_sessions_by_condition(user, ACTIVITY, POSTURE)[3]
    impostor_test_windows.append(load_user_windows(user, [sessions], SENSORS, WINDOW_SIZE, STEP_SIZE))


impostor_test_windows = np.concatenate(impostor_test_windows)

# Checking the shape
print(f'Legit test windows: {legit_test_windows.shape}')
print(f'Impostor test windows: {impostor_test_windows.shape}')

# Normalize - fit on training data only 
mean, std = fit_normalizer(np.concatenate([legit_train_windows, impostor_train_windows]))

legit_train_windows = normalize(legit_train_windows, mean, std)
impostor_train_windows = normalize(impostor_train_windows, mean, std)
legit_test_windows = normalize(legit_test_windows, mean, std)
impostor_test_windows = normalize(impostor_test_windows, mean, std)

# Create datasets
X_train, y_train = create_dataset(legit_train_windows, impostor_train_windows)
X_test, y_test = create_dataset(legit_test_windows, impostor_test_windows)

print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Test Set: {X_test.shape}, {y_test.shape}')

# Initialize model
INPUT_DIM = X_train.shape[2] # 12 features
rnn = VanillaRNN(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM)

print('\n\n')
# Training loop 
for epoch in range(EPOCHS):
    total_loss = 0 
    correct = 0 

    for i in range(len(X_train)):
        # Forward pass 
        a_cache, h_cache = rnn.forward(X_train[i])
        y_pred = rnn.output(h_cache[len(X_train[i]) -1])

        # Compute loss
        loss = rnn.compute_loss(y_train[i], y_pred)
        total_loss += loss[0]

        # Check if prediction is correct
        predicted_label = 1 if y_pred[0] > 0.5 else 0 
        if predicted_label == y_train[i]:
            correct += 1

        # Backward pass 
        grads = rnn.backward(y_train[i], y_pred, h_cache, a_cache, X_train[i])
        grads = rnn.clip_gradients(grads, MAX_NORM)
        rnn.update_weights(grads, LEARNING_RATE)

    avg_loss = total_loss / len(X_train)
    accuracy = correct / len(X_train)
    print(f'Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Evaluate on test set 
print('\n\n')
correct = 0
y_pred_probs = []

for i in range(len(X_test)):
    a_cache, h_cache = rnn.forward(X_test[i])
    y_pred = rnn.output(h_cache[len(X_test[i]) - 1])

    y_pred_probs.append(y_pred)
    predicted_label = 1 if y_pred[0] > 0.5 else 0 

    if predicted_label == y_test[i]:
        correct += 1 

test_accuracy = correct / len(X_test)
print(f'\n Test Accuracy: {test_accuracy:.4f}')
y_pred_probs = np.array(y_pred_probs).flatten()



accuracy, precision, recall, f1, mcc  = evaluate(y_test, y_pred_probs)
eer = compute_eer(y_test, y_pred_probs)


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'MCC: {mcc}')
print(f'EER: {eer}')
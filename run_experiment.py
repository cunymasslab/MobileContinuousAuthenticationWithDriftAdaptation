import numpy as np
from data_loader import load_session_sensors, get_sessions_by_condition
from preprocessor import compute_magnitude, segment_into_windows, fit_normalizer, normalize, create_dataset
from model import VanillaRNN
import math

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

# Computing the EER
def compute_eer(y_true, y_pred_probs):
    '''
    Computes the equal error rate: the threshold where the false accept rate and
    false reject rate intersects.

    Args:
        y_true: a numpy array containing the actual or true values of the datapoints
        y_pred_probs: a numpy array containing the probability at which model predicted those values
    
    Returns:
        the equal error rate
    '''

    # lists to store the far, frr at different thresholds
    far_list = []
    frr_list = []

    # calculating the FAR, FRR at each threshold starting from 1% to 100%
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
    '''
    Computes the accuracy, precision, recall, f1, mcc.
    Uses the confusion_matrix function to get TP, TN, FP, FN

    Args:
        y_true: a numpy array containing the actual or true values of the datapoints
        y_pred_probs: a numpy array containing the probability at which model predicted those values
        threshold: what probability value determines if a prediction is positive or not
    Returns:
        accuracy, precision, recall, f1, mcc
    '''
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred_probs, threshold)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) >0 else 0 
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / denom if denom > 0 else 0 

    return accuracy, precision, recall, f1, mcc

# Confusion Matrix
def confusion_matrix(y_true, y_pred_probs, threshold=0.5):
    '''
    Computes the TP, TN, FP, FN

    Args:
        y_true: a numpy array containing the actual or true values of the datapoints
        y_pred_probs: a numpy array containing the probability at which model predicted those values
        threshold: what probability value determines if a prediction is positive or not
    Returns:
        TP, TN, FP, FN
    '''

    y_pred_labels = (y_pred_probs > threshold).astype(int)
    TP = np.sum((y_pred_labels == 1) & (y_true == 1))
    TN = np.sum((y_pred_labels == 0) & (y_true == 0))
    FP = np.sum((y_pred_labels == 1) & (y_true == 0))
    FN = np.sum((y_pred_labels == 0) & (y_true == 1))
    return TP, TN, FP, FN

def run_experiment(target_user, impostor_users, train_condition, test_condition,
                   algorithm='bptt', sensors=['Accelerometer', 'Gyroscope', 'Magnetometer'],
                   window_size=200,step_size = 200,
                   hidden_dim=32, learning_rate = 0.001,
                   epochs=20, max_norm=1.0):
    '''
    Runs a single authentication experiment.

    Args:
        target_user: the legitmate user of the device
        impostor_users: other users in the dataset chosen to act as impostors
        train_condition: what activity + posture was the model trained on
        test_condition: what activity + posture was the model evaluated on 
        algorithm: bptt or rtrl
        sensors: a combination of acclerometer, magnetometer, gyroscope
        window_size: how many time steps to consider at a given time
        step_size: how much of an overlap of the windows to consider
        hidden_dim: how many neurons in the models' hidden layer 
        learning_rate: how fast the model is learning
        epochs: how many full passes through the dataset 
        max_norm: parameter used for graident clipping
    
    Returns:
        dictionary with keys: 
          'target_user',
                'train_condition',
                'test_condition',
                'algorithm',
                'accuracy',
                'precision',
                'recall',
                'f1',
                'mcc',
                'eer',

    '''
    
    np.random.seed(42)

    train_activity, train_posture = train_condition.split('+')
    test_activity, test_posture = test_condition.split('+')

    # loading the data
    train_session_numbers = get_sessions_by_condition(target_user, train_activity, train_posture)

    # separating training and test sessions

    # using the first 3 sessions as training data
    train_sessions = train_session_numbers[:3]

    # using the last  session as test data 
    test_session_numbers = get_sessions_by_condition(target_user, test_activity, test_posture)
    test_sessions = test_session_numbers[3]


    # getting the training data for the legit user
    legit_train_windows = load_user_windows(target_user, train_sessions, sensors, window_size, step_size)

    # getting the training data for the impostor
    impostor_train_windows = []

    for user in impostor_users:
        sessions = get_sessions_by_condition(user, train_activity, train_posture)[:3]
        impostor_train_windows.append(load_user_windows(user, sessions, sensors, window_size, step_size))

    impostor_train_windows = np.concatenate(impostor_train_windows)

    # downsampling
    if len(impostor_train_windows) > len(legit_train_windows):
        indices = np.random.choice(len(impostor_train_windows), size = len(legit_train_windows), replace= False)
        impostor_train_windows = impostor_train_windows[indices]

    # Checking the shape of the training data 
    print(f'Legit Train Windows Shape: {legit_train_windows.shape}')
    print(f'Impostor Train Windows Shape: {impostor_train_windows.shape}')

    # Loading the Test Data 

    # Getting the test data for the legit users
    legit_test_windows = load_user_windows(target_user, [test_sessions], sensors, window_size, step_size)

    # Getting the test data for the impostor users
    impostor_test_windows = []

    for user in impostor_users:
        sessions = get_sessions_by_condition(user, test_activity, test_posture)[3]
        impostor_test_windows.append(load_user_windows(user, [sessions], sensors, window_size, step_size))

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

     # Initialize model
    input_dim = X_train.shape[2]
    rnn = VanillaRNN(input_dim=input_dim, hidden_dim=hidden_dim)

     # Training loop
    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for i in range(len(X_train)):
            if algorithm == 'bptt':
                a_cache, h_cache = rnn.forward(X_train[i])
                h_final = h_cache[len(X_train[i]) - 1]
                y_pred = rnn.output(h_final)
                loss = rnn.compute_loss(y_train[i], y_pred)
                total_loss += loss[0]
                grads = rnn.backward(y_train[i], y_pred, h_cache, a_cache, X_train[i])

            elif algorithm == 'rtrl':
                h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward(X_train[i])
                y_pred = rnn.output(h_final)
                loss = rnn.compute_loss(y_train[i], y_pred)
                total_loss += loss[0]
                grads = rnn.rtrl_compute_gradients(y_train[i], y_pred, h_final, S_xh, S_hh, S_bh)

            predicted_label = 1 if y_pred[0] > 0.5 else 0
            if predicted_label == y_train[i]:
                correct += 1

            grads = rnn.clip_gradients(grads, max_norm)
            rnn.update_weights(grads, learning_rate)

        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

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

    return {
                'target_user': target_user,
                'train_condition': train_condition,
                'test_condition': test_condition,
                'algorithm': algorithm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc,
                'eer': eer
            }



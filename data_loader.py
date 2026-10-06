"""
data_loader.py
Loads raw HMOG sensor data from disk. Handles file I/O and sensor combination.
Knows about file paths, CSV formats, and session metadata. Outputs raw numpy arrays.
"""

from dotenv import load_dotenv 
import os 
import numpy as np

# Loading the environment variables 
load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

def load_sensor_file(file_path):
    """
    Reads a single HMOG sensor CSV file (no headers, 7 columns).
    Returns the full numpy array (always 2D).
    """
    data = np.loadtxt(file_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def load_session_sensors(user_id, session_number, sensors):
    """
    Loads and combines x, y, z columns from multiple sensor files for one session.
    
    Args:
        user_id: string, e.g. '100669'
        session_number: int, e.g. 1
        sensors: list of sensor names, e.g. ['Accelerometer', 'Gyroscope']
    
    Returns:
        numpy array of shape (T, 3 * num_sensors) where T is the number of timesteps
    """
        
    all_data = []

    for sensor in sensors:
        file_path = os.path.join(dataset_path, user_id, user_id + '_session_' + str(session_number), sensor + '.csv')
        raw = load_sensor_file(file_path)
        xyz = raw[:, 3:6 ]
        all_data.append(xyz)
    
    min_length = min(arr.shape[0] for arr in all_data)
    all_data = [arr[:min_length] for arr in all_data]

    return np.hstack(all_data)

def get_taskID(taskID):
    """
    Maps a TaskID number to its activity+posture string. 

    Args:
        taskID: int, TaskID from Activity.csv (1-24)

    Returns:
        string like 'READING+SITTING' or 'WRITING+WALKING'
    """

    taskID_map = {
        1 : 'READING+SITTING',
        2 : 'READING+WALKING',
        3 : 'WRITING+SITTING',
        4 : 'WRITING+WALKING',
        5 : 'MAP+SITTING',
        0 : 'MAP+WALKING'
    }

    return taskID_map[taskID % 6]

def get_session_activity_info(user_id, session_number):
    """
    Reads Activity.csv for a session and returns the activity+posture condition.
    
    Args:
        user_id: string, e.g. '100669'
        session_number: int, 1-24
    
    Returns:
        string like 'READING+SITTING'
    """
    file_path = os.path.join(dataset_path,user_id, user_id + '_session_' + str(session_number), 'Activity.csv')
    raw = load_sensor_file(file_path)
    task_id = int(raw[0, 8])
    return get_taskID(task_id)

def get_sessions_by_condition(user_id, activity, posture):
    """
    Finds all session numbers for a user that match a given activity and posture.
    
    Args:
        user_id: string, e.g. '100669'
        activity: string, one of 'READING', 'WRITING', 'MAP'
        posture: string, one of 'SITTING', 'WALKING'
    
    Returns:
        list of session numbers (ints) matching the condition
    """
    matching_sessions = []

    for session_number in range(1, 25):
        session_dir = os.path.join(dataset_path, user_id, user_id + '_session_' + str(session_number))
        activity_file = os.path.join(session_dir, 'Activity.csv')
        
        if not os.path.isdir(session_dir) or not os.path.isfile(activity_file):
            continue

        activity_found, posture_found = get_session_activity_info(user_id, session_number).split('+')

        if activity == activity_found and posture == posture_found:
            matching_sessions.append(session_number)

    return matching_sessions


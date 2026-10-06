import csv
import os
from dotenv import load_dotenv
from run_experiment import run_experiment
from data_loader import get_session_activity_info
import random 
import time
import numpy as np
import sys

# importing environment variables
load_dotenv()

dataset_path = os.getenv('HMOG_ROOT')

# Setting random state for reproducibility
np.random.seed(42)
random.seed(42)



# selecting 'n' random users to be impostors
def get_impostors(target_user, all_users, n =10):
    candidates = [u for u in all_users if u != target_user]
    return random.sample(candidates, min(n, len(candidates)))

# Get valid users (have at least 1 session)
all_entries = os.listdir(dataset_path)
valid_users = []

for d in sorted(all_entries):
    user_path = os.path.join(dataset_path, d)

    if not os.path.isdir(user_path):
        continue

    session_dirs = [s for s in os.listdir(user_path) if s.startswith(d + '_session_')]

    if len(session_dirs) > 0:
        valid_users.append(d)

print(f'Found {len(valid_users)} valid users')

# taking command line args
algorithm_arg = sys.argv[1] if len(sys.argv) > 1 else 'bptt'
hidden_dim_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 32
num_users = int(sys.argv[3]) if len(sys.argv) > 3 else len(valid_users)
mode = sys.argv[4] if len(sys.argv) > 4 else 'level1'
output_file = f'results_{algorithm_arg}_h{hidden_dim_arg}_{mode}.csv'


LEVEL2_PAIRS = [
    ('READING+SITTING',  'READING+WALKING'),
    ('WRITING+SITTING',  'WRITING+WALKING'),
    ('MAP+SITTING',      'MAP+WALKING'),
    ('READING+SITTING',  'WRITING+SITTING'),
    ('READING+SITTING',  'MAP+SITTING'),
]



CONDITIONS = [
    'READING+SITTING', 'READING+WALKING',
    'WRITING+SITTING', 'WRITING+WALKING',
    'MAP+SITTING', 'MAP+WALKING'
]

ALGORITHMS = ['bptt', 'rtrl']

fieldnames = ['target_user', 'train_condition', 'test_condition', 
              'algorithm', 'accuracy', 'precision', 'recall', 
              'f1', 'mcc', 'eer', 'runtime_seconds']

# Resume support: if file exists, skip header and track completed experiments
completed = set()
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row['target_user'], row['train_condition'], row['test_condition'], row['algorithm']))
    print(f'Resuming: {len(completed)} experiments already completed')
else:
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Setting up train test pairs based on experiment level
if mode == 'level1':
    pairs = [(c, c) for c in CONDITIONS]
elif mode == 'level2':
    pairs = LEVEL2_PAIRS

# Run experiments
for user in valid_users[:num_users]:
    impostors = get_impostors(user, valid_users)

    for train_condition, test_condition in pairs:
        for algorithm in [algorithm_arg]:

            print(f'\n=== User: {user}, Train Condition: {train_condition}, Test Condition: {test_condition}, Algorithm: {algorithm} ===')

            # Skip if already completed
            if (user, train_condition, test_condition, algorithm) in completed:
                print('Already completed, skipping.')
                continue

            start = time.time()
            
            try:
                result = run_experiment(target_user=user, impostor_users=impostors, train_condition=train_condition, test_condition=test_condition, algorithm=algorithm, hidden_dim=hidden_dim_arg)
                result['runtime_seconds'] = time.time() - start

                # Append result to CSV after each experiment
                with open(output_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)
                print(result)
            except Exception as e:
                print(f'FAILED: User {user}, {train_condition}/{test_condition}, {algorithm}: {e}')
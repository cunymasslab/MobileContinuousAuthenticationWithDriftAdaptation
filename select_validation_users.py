"""
select_validation_users.py
Randomly selects 10 validation users and saves 90 test users.
Run once before any hyperparameter search. Lock in permanently.
"""

import os
import random
from dotenv import load_dotenv

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

all_entries = os.listdir(dataset_path)
valid_users = []
for d in sorted(all_entries):
    user_path = os.path.join(dataset_path, d)
    if not os.path.isdir(user_path):
        continue
    session_dirs = [s for s in os.listdir(user_path) if s.startswith(d + '_session_')]
    if len(session_dirs) > 0:
        valid_users.append(d)

print(f'Total valid users: {len(valid_users)}')

random.seed(42)
validation_users = sorted(random.sample(valid_users, 10))
test_users = sorted(u for u in valid_users if u not in validation_users)

with open('validation_users.txt', 'w') as f:
    for u in validation_users:
        f.write(u + '\n')

with open('test_users.txt', 'w') as f:
    for u in test_users:
        f.write(u + '\n')

print(f'Validation users ({len(validation_users)}): {validation_users}')
print(f'Test users ({len(test_users)}): {test_users}')
print('Saved to validation_users.txt and test_users.txt')

from dotenv import load_dotenv
import os
from data_loader import get_session_activity_info

# Loading the environment variables 
load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

all_entries = os.listdir(dataset_path)

user_dirs = [ d for d in all_entries if os.path.isdir(os.path.join(dataset_path, d))]

print(f'Found {len(user_dirs)} users')

print(user_dirs[:5])

for user in sorted(user_dirs):
    # list session folders for this user 
    user_path  = os.path.join(dataset_path, user)

    # find all session folders for this user (e.g., '100669_session_1', '100669_session_2')
    session_dirs = [d for d in os.listdir(user_path) if d.startswith(user + '_session_')]

    # skip users with no sessions (e.g. empty folders)
    if len(session_dirs) == 0:
        continue

    # count how many sessions belong to each condition (e.g. READING_SITTING: 4)
    condition_counts = {}

    for sd in session_dirs:
        # extract session number from folder name (e.g., '100669_session_14' -> 14)
        session_num = int(sd.split('_session_')[1])

        # look up the activity+posture for this session using Activity.csv
        condition = get_session_activity_info(user, session_num)

        # increment the count for this condition 
        condition_counts[condition] = condition_counts.get(condition, 0) + 1

    print(f'\nUser {user}: {len(session_dirs)} sessions')

    for condition, count in sorted(condition_counts.items()):
        print(f' {condition}: {count} sessions')

    

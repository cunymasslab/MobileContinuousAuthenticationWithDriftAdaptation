"""
split_chunks.py
Splits test users into N non-overlapping chunks for parallel Stage 4 runs.
Usage: python split_chunks.py [--num-chunks 4]
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Split test users into chunks')
    parser.add_argument('--num-chunks', type=int, default=4)
    parser.add_argument('--users-file', type=str, default='test_users.txt')
    args = parser.parse_args()

    if not os.path.exists(args.users_file):
        print(f'ERROR: {args.users_file} not found.')
        return

    with open(args.users_file) as f:
        users = sorted([line.strip() for line in f if line.strip()])

    print(f'Total users: {len(users)}')

    for i in range(args.num_chunks):
        chunk_size = len(users) // args.num_chunks
        remainder = len(users) % args.num_chunks
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_users = users[start:end]

        filename = f'chunk_{i+1}_users.txt'
        with open(filename, 'w') as f:
            for u in chunk_users:
                f.write(u + '\n')
        print(f'  {filename}: {len(chunk_users)} users ({chunk_users[0]} - {chunk_users[-1]})')

    # Verify non-overlapping
    all_chunk_users = set()
    for i in range(args.num_chunks):
        with open(f'chunk_{i+1}_users.txt') as f:
            chunk = set(line.strip() for line in f if line.strip())
        overlap = all_chunk_users & chunk
        if overlap:
            print(f'ERROR: chunk_{i+1} overlaps with previous chunks: {overlap}')
            return
        all_chunk_users |= chunk

    if all_chunk_users == set(users):
        print(f'Verified: {args.num_chunks} chunks, no overlaps, all {len(users)} users covered.')
    else:
        missing = set(users) - all_chunk_users
        print(f'ERROR: {len(missing)} users missing from chunks: {missing}')


if __name__ == '__main__':
    main()
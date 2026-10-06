"""
merge_results.py
Merges Stage 4 chunk CSV files into a single results file.
Usage: python merge_results.py [--num-chunks 4] [--stage stage4]
"""

import argparse
import csv
import os


def merge_csvs(pattern, num_chunks, output_file):
    """Merge chunk CSVs into one file, with validation."""
    all_rows = []
    all_users = set()

    for i in range(1, num_chunks + 1):
        filename = pattern.format(chunk=i)
        if not os.path.exists(filename):
            print(f'  WARNING: {filename} not found — skipping')
            continue

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            chunk_rows = list(reader)

        chunk_users = set(row.get('target_user', '') for row in chunk_rows)
        chunk_users.discard('')

        print(f'  {filename}: {len(chunk_rows)} rows, {len(chunk_users)} unique users')

        # Check for cross-chunk user overlap
        overlap = all_users & chunk_users
        if overlap:
            print(f'  WARNING: {len(overlap)} users appear in multiple chunks: '
                  f'{sorted(overlap)[:5]}...')

        all_users |= chunk_users
        all_rows.extend(chunk_rows)

    if len(all_rows) == 0:
        print('ERROR: No data found across any chunks.')
        return

    # Write merged file
    # Collect all fieldnames across all chunks
    all_fieldnames = []
    seen = set()
    for row in all_rows:
        for key in row.keys():
            if key not in seen:
                all_fieldnames.append(key)
                seen.add(key)
    fieldnames = all_fieldnames

    # Deduplicate (crash-resume can produce duplicate detail rows)
    if 'window_index' in fieldnames:  # detail CSV
        seen_keys = set()
        deduped = []
        for row in all_rows:
            key = (row.get('target_user'), row.get('train_condition'),
                   row.get('stream_condition'), row.get('seed'), row.get('window_index'))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(row)
        if len(deduped) < len(all_rows):
            print(f'  Removed {len(all_rows) - len(deduped)} duplicate rows')
            all_rows = deduped

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)



    print(f'\nMerged: {len(all_rows)} total rows, {len(all_users)} unique users -> {output_file}')

    # Warn if low coverage
    expected_users = 90
    if len(all_users) < expected_users * 0.9:
        print(f'  WARNING: Only {len(all_users)}/{expected_users} users '
              f'({100*len(all_users)/expected_users:.0f}% coverage)')


def main():
    parser = argparse.ArgumentParser(description='Merge Stage 4 chunk results')
    parser.add_argument('--num-chunks', type=int, default=4)
    parser.add_argument('--stage', type=str, default='stage4')
    args = parser.parse_args()

    stage = args.stage

    print(f'=== Merging {stage} summary CSVs ===')
    merge_csvs(
        pattern=f'results_{stage}_chunk{{chunk}}_summary.csv',
        num_chunks=args.num_chunks,
        output_file=f'results_{stage}_merged_summary.csv',
    )

    print(f'\n=== Merging {stage} detail CSVs ===')
    merge_csvs(
        pattern=f'results_{stage}_chunk{{chunk}}_detail.csv',
        num_chunks=args.num_chunks,
        output_file=f'results_{stage}_merged_detail.csv',
    )

    print(f'\n=== Merging {stage} training curves ===')
    merge_csvs(
        pattern=f'results_{stage}_chunk{{chunk}}_training_curves.csv',
        num_chunks=args.num_chunks,
        output_file=f'results_{stage}_merged_training_curves.csv',
    )


if __name__ == '__main__':
    main()
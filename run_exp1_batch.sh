#!/bin/bash
# run_exp1_batch.sh
# Runs run_stage4b_exp1.py on the 50 user-condition assignments in exp1_50_users.txt

USER_LIST="exp1_50_users.txt"
OUTPUT_DIR="results_stage4b_exp1"
LOG_FILE="exp1_batch.log"
SEED=42

mkdir -p "$OUTPUT_DIR"
echo "Batch started: $(date)" >> "$LOG_FILE"

while IFS=',' read -r user train_cond stream_cond; do
    # Skip blank lines and comments
    [ -z "$user" ] && continue
    [[ "$user" == \#* ]] && continue
    
    echo "=== $(date) | user=$user train=$train_cond stream=$stream_cond ===" >> "$LOG_FILE"
    python3 run_stage4b_exp1.py \
        --user "$user" \
        --train-condition "$train_cond" \
        --stream-condition "$stream_cond" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "FAILED: user=$user train=$train_cond stream=$stream_cond" >> "$LOG_FILE"
    fi
done < "$USER_LIST"

echo "Batch finished: $(date)" >> "$LOG_FILE"
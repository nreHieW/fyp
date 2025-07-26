#!/bin/bash

set -e

FILES=(

)


for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    if [[ "$BASENAME" =~ [0-9]+\.jsonl$ ]]; then
        echo "Running: uv run evaluate.py --eval_similarity --sample_path $FILE"
        uv run partial_edits/evaluate.py --eval_similarity --sample_path "$FILE"
    else
        echo "Running: uv run evaluate.py --eval_similarity --sample_path $FILE --hard"
        uv run partial_edits/evaluate.py --eval_similarity --sample_path "$FILE" --hard
    fi
    # echo "Running: uv run evaluate.py --eval_similarity --sample_path $FILE"
    # uv run partial_edits/evaluate.py --eval_similarity --sample_path "$FILE"
done

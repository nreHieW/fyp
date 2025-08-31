#!/bin/bash

set -e

FILES=(
)

# Check if all files exist before proceeding
MISSING_FILES=()

for FILE in "${FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        MISSING_FILES+=("$FILE")
    fi
done

# Print all missing files if any
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "Error: The following files are missing:"
    for FILE in "${MISSING_FILES[@]}"; do
        echo "  - $FILE"
    done
    echo "Total missing files: ${#MISSING_FILES[@]}"
    exit 1
fi

for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    
    # Check if filename contains "140" or doesn't end with any number
    if [[ "$BASENAME" =~ 140 ]] || [[ ! "$BASENAME" =~ [0-9]+\.jsonl$ ]]; then
        echo "Running: uv run -m partial_edits.evaluate --eval_similarity --sample_path $FILE --hard"
        uv run -m partial_edits.evaluate --eval_similarity --sample_path "$FILE" --hard
    else
        echo "Running: uv run -m partial_edits.evaluate --eval_similarity --sample_path $FILE"
        uv run -m partial_edits.evaluate --eval_similarity --sample_path "$FILE"
    fi
done

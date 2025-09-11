MODEL_LIST="
nreHieW/partial-edits-sft-qwen3-4b-instruct-2507
"

PORT=8000
DATA_PARALLEL=4
QUESTIONS_PATH=data/questions/corrupted_solutions_manual_easy_400.jsonl
RELEASE_VERSION=v6
N=1

for MODEL in $MODEL_LIST; do
echo "=== Starting vLLM for $MODEL on port $PORT ==="

# Ensure port is free
if nc -z localhost $PORT >/dev/null 2>&1; then
echo "Port $PORT is already in use. Please free it and retry."
exit 1
fi

# Launch vLLM in a new Terminal window and capture the window id
WINDOW_ID=$(osascript -e 'tell application "Terminal" to activate' \
-e "tell application \"Terminal\" to set t to do script \"vllm serve $MODEL --data-parallel-size $DATA_PARALLEL --port $PORT\"" \
-e 'tell application "Terminal" to set wid to id of window of t' \
-e 'tell application "Terminal" to return wid')

if [ -z "$WINDOW_ID" ]; then
echo "Failed to open Terminal window for vLLM."
exit 1
fi

# Wait for server readiness (timeout ~120s)
ATTEMPTS=0
until [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models)" = "200" ] || [ $ATTEMPTS -ge 60 ]; do
sleep 2
ATTEMPTS=$((ATTEMPTS+1))
done

if [ $ATTEMPTS -ge 60 ]; then
echo "vLLM server did not become ready on port $PORT. Cleaning up..."
pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
osascript <<EOF
tell application "Terminal"
try
if exists (window id $WINDOW_ID) then close (window id $WINDOW_ID)
end try
end tell
EOF
exit 1
fi

echo "=== Running generator against local/$MODEL ==="
uv run partial_edits/generate_solutions.py --questions_path "$QUESTIONS_PATH" --store_token_info --model "local/$MODEL"

echo "=== Stopping vLLM and closing its Terminal window ==="
pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
sleep 2
osascript <<EOF
tell application "Terminal"
try
if exists (window id $WINDOW_ID) then close (window id $WINDOW_ID)
end try
end tell
EOF

echo "=== Running LCB evaluation for $MODEL ==="
uv run python -m lcb_runner.runner.main --model "$MODEL" --scenario codegeneration --evaluate --release_version "$RELEASE_VERSION" --n "$N"

echo "=== Completed $MODEL ===\n"
done
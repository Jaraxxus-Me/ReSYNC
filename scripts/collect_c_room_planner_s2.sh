#!/bin/bash
# Script to run data collection with automatic restart on failure
# Leverages incremental saving to resume from checkpoints

set -u  # Exit on undefined variables

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT" || exit 1

PYTHON_SCRIPT="$SCRIPT_DIR/collect_c_room_planner_s2.py"
ATTEMPT=0
MAX_ATTEMPTS=100  # Safety limit to prevent infinite loops

echo "========================================="
echo "Starting robust data collection"
echo "Script: $PYTHON_SCRIPT"
echo "Will retry on failure (max $MAX_ATTEMPTS attempts)"
echo "========================================="
echo

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "----------------------------------------"
    echo "Attempt #$ATTEMPT - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"

    # Run the Python script
    python "$PYTHON_SCRIPT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo
        echo "========================================="
        echo "SUCCESS! Data collection completed"
        echo "Total attempts: $ATTEMPT"
        echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="
        exit 0
    else
        echo
        echo "⚠ Attempt #$ATTEMPT failed with exit code $EXIT_CODE"
        echo "Restarting in 5 seconds (will resume from checkpoint)..."
        echo
        sleep 5
    fi
done

echo
echo "========================================="
echo "ERROR: Maximum attempts ($MAX_ATTEMPTS) reached"
echo "Data collection did not complete successfully"
echo "========================================="
exit 1

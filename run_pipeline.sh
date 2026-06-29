#!/bin/bash

# Exit on error
set -e

# Groq API key — set via environment variable (do NOT hardcode here)
# export GROQ_API_KEY="your_key_here"  # or set in your shell profile / .env file
if [ -z "$GROQ_API_KEY" ]; then
    echo "Warning: GROQ_API_KEY is not set. NLP queries will be skipped."
fi

# Video file to process (defaults to test_seated3.mkv if none specified)
VIDEO_PATH="${1:-test_seated3.mkv}"
CAMERA_NAME="cam_test"

echo "=========================================="
echo "         RESTAURANT ANALYTICS RUNNER       "
echo "=========================================="
echo "Processing video: $VIDEO_PATH"
echo "=========================================="

# 1. Clear database
echo "--> 1. Clearing previous database entries..."
./venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('db/customer_intel.db')
conn.execute('DELETE FROM wait_metrics')
conn.execute('DELETE FROM persons')
conn.commit()
conn.close()
print('Database tables (persons, wait_metrics) successfully cleared.')
"

# 2. Update pipeline_position.py video configuration
echo "--> 2. Configuring pipeline_position.py..."
VIDEO_PATH="$VIDEO_PATH" CAMERA_NAME="$CAMERA_NAME" ./venv/bin/python -c "
import os, re
video_path = os.environ['VIDEO_PATH']
camera_name = os.environ['CAMERA_NAME']
with open('pipeline_position.py', 'r') as f:
    content = f.read()

# Replace VIDEOS = [ ... ] list dynamically
pattern = r'VIDEOS = \[[^\]]*\]'
replacement = f\"VIDEOS = [('{video_path}', '{camera_name}')]\"
updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('pipeline_position.py', 'w') as f:
    f.write(updated_content)
print(f'Successfully targeted pipeline_position.py to: {video_path}')
"

# 3. Run the camera tracking pipeline
echo "--> 3. Launching tracking pipeline..."
./venv/bin/python pipeline_position.py

# 4. Run question answering agent
echo "=========================================="
echo "        QUESTION ANSWERING SYSTEM         "
echo "=========================================="

if [ -z "$GROQ_API_KEY" ]; then
    echo "Note: GROQ_API_KEY environment variable is not set."
    echo "To run the Natural Language query agent, set your key first:"
    echo "  export GROQ_API_KEY='your_api_key'"
    echo "Then run the query agent manual command:"
    echo "  ./venv/bin/python agent/query_agent.py"
else
    echo "Running query agent on the database records..."
    ./venv/bin/python agent/query_agent.py
fi

echo "=========================================="
echo "Run complete."
echo "=========================================="

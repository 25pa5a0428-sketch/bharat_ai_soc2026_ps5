#!/bin/bash
set -e
echo "=== STAGE 1: YOLO ==="
python3 /home/nano/run_batch.py
echo "=== STAGE 2: OFFLINE TINYLLAMA ==="
sudo docker start pcb_llama >/dev/null
sudo docker exec pcb_llama python3 /workspace/run_llama_advisor.py

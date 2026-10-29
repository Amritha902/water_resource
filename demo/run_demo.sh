#!/bin/sh
# Start the live demo:  sh demo/run_demo.sh
cd "$(dirname "$0")/.." && python3 demo/run_demo.py "$@"

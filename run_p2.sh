#!/bin/bash
# P2: arms 1,3 in-process; arm 2 via mlx_lm.server (nonce = cache-busted, then identical prompts = cached).
cd "$(dirname "$0")"; PY=.venv-mlxlm/bin/python
$PY bench.py --arm 1 > results-arm1.log 2>&1
$PY bench.py --arm 3 > results-arm3.log 2>&1
$PY -m mlx_lm.server --model models/rapid-mlx-4bit --port 8080 > server.log 2>&1 &
SP=$!; sleep 15
$PY bench.py --arm 2 --server-pid $SP --server-version "mlx-lm 0.32.2" --model-label rapid-mlx-4bit > results-arm2.log 2>&1
$PY bench.py --arm 2 --server-pid $SP --server-version "mlx-lm 0.32.2" --model-label rapid-mlx-4bit --no-nonce --tag cached --prompts short medium long > results-arm2c.log 2>&1
kill $SP; echo DONE > p2.done

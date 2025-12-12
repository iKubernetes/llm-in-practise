#!/bin/bash
set -e

ray stop --force || true
pkill -9 ray raylet || true

ray start \
  --address='172.25.0.100:6379' \
  --node-ip-address=172.25.0.200 \
  --min-worker-port=12000 \
  --max-worker-port=13000 \
  --disable-usage-stats

echo "Ray Worker started and joined Head at 172.25.0.100"


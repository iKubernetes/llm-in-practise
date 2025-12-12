#!/bin/bash
set -e

ray stop --force || true
pkill -9 ray raylet || true

ray start \
  --head \
  --node-ip-address=172.25.0.100 \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --min-worker-port=12000 \
  --max-worker-port=13000 \
  --disable-usage-stats

echo "Ray Head started on 172.25.0.100"
echo "Dashboard: http://172.25.0.100:8265"

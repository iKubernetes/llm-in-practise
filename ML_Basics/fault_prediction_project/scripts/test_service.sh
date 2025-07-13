#!/bin/bash

# 测试样本
SAMPLES=(
  '{"timestamp": "2025-07-13 04:25:00", "device_id": "server_001", "cpu_usage": 90.0, "ram_usage": 85.0, "disk_io": 50.0, "temperature": 45.0, "error_count": 10}'
  '{"timestamp": "2025-07-13 04:26:00", "device_id": "server_002", "cpu_usage": 50.0, "ram_usage": 60.0, "disk_io": 20.0, "temperature": 35.0, "error_count": 2}'
  '{"timestamp": "2025-07-13 04:27:00", "device_id": "server_001", "cpu_usage": 5.0, "ram_usage": 10.0, "disk_io": 2.0, "temperature": 32.0, "error_count": 8}'
)

# 发送请求
for sample in "${SAMPLES[@]}"; do
  echo "Testing sample: $sample"
  curl -X POST http://localhost:5000/predict_fault \
       -H "Content-Type: application/json" \
       -d "$sample"
  echo -e "\n"
done

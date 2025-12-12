#!/bin/bash
set -e

echo "正在彻底清理 Ray 所有遗留..."

ray stop --force || true
pkill -9 -f ray:: || true
pkill -9 -f raylet || true

rm -rf /tmp/ray /tmp/raylogs /tmp/ray_ssh_* ~/.ray ~/ray_results ~/.cache/ray
sudo rm -rf /tmp/ray* /tmp/raylogs* /var/tmp/ray*
sudo rm -f /dev/shm/ray_* /var/run/ray_*

sudo systemctl stop ray-* 2>/dev/null || true
sudo systemctl disable ray-* 2>/dev/null || true
sudo rm -f /etc/systemd/system/ray-*.service
sudo systemctl daemon-reload

echo "Ray 已彻底清理干净，当前状态等同于从未安装过"
ray --version 2>/dev/null || echo "ray 命令已不可用（正常）"

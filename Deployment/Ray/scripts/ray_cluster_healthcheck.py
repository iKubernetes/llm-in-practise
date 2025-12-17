"""
Ray Cluster 综合健康检查脚本

验证内容：
1. Driver 是否能成功连接 Ray Cluster
2. Worker Process 是否能在多个节点正常启动
3. CPU 调度是否正常
4. GPU（若存在）是否可被 Ray 识别并使用
5. Object Store（Plasma）跨节点数据通路是否正常

适用场景：
- 裸机 / VM Ray Cluster
- Docker / 容器化 Ray Cluster
- Kubernetes（KubeRay）集群
"""

import ray
import socket
import os
import time
import subprocess
import numpy as np
from collections import defaultdict

# ---------------------------------------------------
# 1. 连接到已存在的 Ray Cluster
# ---------------------------------------------------
# address="auto" 表示：
# - 从环境变量 / Ray Client / 本地配置中自动发现 Head Node
# - 如果连接失败，说明 Driver ↔ Head 网络或 Ray 未启动
ray.init(address="auto")

# ---------------------------------------------------
# 2. 获取并打印集群基础信息
# ---------------------------------------------------
# ray.nodes() 返回的是 Head Node 维护的全局节点视图
# Alive=True 表示该节点当前仍在集群中
nodes = [n for n in ray.nodes() if n["Alive"]]

print(f"\n=== Ray Cluster Health Check ===")
print(f"Alive nodes: {len(nodes)}")

# 打印每个节点的 CPU / GPU 资源视图
# 这是 Ray 调度器实际看到的资源，不等同于物理机器资源
for n in nodes:
    print(
        f" - {n['NodeManagerAddress']} "
        f"| CPUs={n['Resources'].get('CPU', 0)} "
        f"| GPUs={n['Resources'].get('GPU', 0)}"
    )

# ---------------------------------------------------
# 3. CPU Worker Process 测试
# ---------------------------------------------------
# 目的：
# - 验证 Worker Process 是否能被正常拉起
# - 验证 Ray 调度器是否能把 Task 分发到不同节点
#
# 注意：
# - 不绑定 node_id
# - 依赖 Ray 默认的调度与负载均衡行为
@ray.remote
def cpu_probe():
    return {
        "type": "cpu",
        # hostname 用于判断 Task 实际运行在哪台机器上
        "hostname": socket.gethostname(),
        # pid 用于确认这是独立的 Worker Process
        "pid": os.getpid(),
        # Ray 内部的 node_id，用于精确区分节点
        "node_id": ray.get_runtime_context().get_node_id(),
    }

# 每个节点发多个 Task，避免所有 Task 被调度到同一个节点
cpu_tasks = [cpu_probe.remote() for _ in range(len(nodes) * 2)]
cpu_results = ray.get(cpu_tasks)

# 统计 CPU Task 实际覆盖的节点数量
cpu_hosts = {r["hostname"] for r in cpu_results}

print(f"\n[CPU] Tasks ran on {len(cpu_hosts)} unique hosts")
for r in cpu_results:
    print(r)

# ---------------------------------------------------
# 4. GPU Worker Process 测试（如果集群中存在 GPU）
# ---------------------------------------------------
# ray.available_resources() 返回当前集群可用资源的聚合视图
# 如果 GPU = 0，说明：
# - Worker 启动时未声明 GPU
# - 或容器内 GPU 不可见
total_gpus = int(ray.available_resources().get("GPU", 0))
print(f"\n[GPU] Total GPUs visible to Ray: {total_gpus}")

gpu_results = []

if total_gpus > 0:

    # num_gpus=1 表示：
    # - 每个 Task 独占 1 张 GPU
    # - Ray 会自动设置 CUDA_VISIBLE_DEVICES
    @ray.remote(num_gpus=1)
    def gpu_probe():
        try:
            # 使用 nvidia-smi 验证 CUDA / Driver 是否可用
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                text=True
            )
        except Exception as e:
            smi = f"ERROR: {e}"

        return {
            "type": "gpu",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_info": smi.strip(),
        }

    # 创建与 GPU 数量相同的 Task
    # 期望所有 GPU 都能被调度并使用
    gpu_tasks = [gpu_probe.remote() for _ in range(total_gpus)]
    gpu_results = ray.get(gpu_tasks)

    gpu_hosts = {r["hostname"] for r in gpu_results}
    print(f"[GPU] Tasks ran on {len(gpu_hosts)} unique hosts")

    for r in gpu_results:
        print(r)

else:
    print("[GPU] Skipped (no GPU in cluster)")

# ---------------------------------------------------
# 5. Object Store（Plasma）数据通路测试
# ---------------------------------------------------
# 目的：
# - 验证 ray.put / ray.get 是否正常
# - 验证大对象是否能在多个 Worker 之间共享
# - 验证 Object Store /dev/shm 是否可用
@ray.remote
def object_consumer(arr):
    return {
        "hostname": socket.gethostname(),
        # 对数组求和，验证数据完整性
        "sum": float(arr.sum()),
    }

print("\n[Object Store] Testing data sharing...")

# 构造一个较大的 numpy 数组
# 会被放入 Plasma Object Store
big_array = np.ones((2048, 2048), dtype=np.float32)

# 将对象放入 Object Store
obj_ref = ray.put(big_array)

# 在多个 Worker 上消费同一个对象引用
obj_tasks = [object_consumer.remote(obj_ref) for _ in range(len(nodes))]
obj_results = ray.get(obj_tasks)

obj_hosts = {r["hostname"] for r in obj_results}
print(f"[Object Store] Data consumed on {len(obj_hosts)} unique hosts")

for r in obj_results:
    print(r)

# ---------------------------------------------------
# 6. 最终健康检查汇总
# ---------------------------------------------------
print("\n=== Health Check Summary ===")
print(f"CPU hosts: {len(cpu_hosts)}")
print(f"GPU hosts: {len({r['hostname'] for r in gpu_results}) if gpu_results else 0}")
print(f"Object Store hosts: {len(obj_hosts)}")

print("\nRay Cluster health check completed successfully.")

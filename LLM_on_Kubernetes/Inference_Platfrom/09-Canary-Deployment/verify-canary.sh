#!/bin/bash
# verify-canary.sh
NAMESPACE="llm-inference"
ROLLOUT="vllm-qwen3-8b"

echo "=== 1. Rollout 状态 ==="
kubectl argo rollouts get rollout $ROLLOUT -n $NAMESPACE

echo -e "\n=== 2. Ingress Canary 权重 ==="
kubectl get ingress vllm-canary -n $NAMESPACE -o jsonpath='{.metadata.annotations.nginx\.ingress\.kubernetes\.io/canary-weight}'

echo -e "\n\n=== 3. Endpoints 分布 ==="
echo "Stable:"
kubectl get endpoints vllm-stable -n $NAMESPACE
echo "Canary:"
kubectl get endpoints vllm-canary -n $NAMESPACE

echo -e "\n=== 4. Pod 版本分布 ==="
kubectl get pods -n $NAMESPACE -l app=vllm-qwen3-8b -o custom-columns=NAME:.metadata.name,IMAGE:.spec.containers[0].image,STATUS:.status.phase

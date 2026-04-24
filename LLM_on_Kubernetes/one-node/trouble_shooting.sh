echo "=== Ingress ==="
kubectl get ingress qwen3-8b-ingress -o yaml

echo "=== Service ==="
kubectl get svc qwen3-8b-service -o yaml

echo "=== Endpoints ==="
kubectl get endpoints qwen3-8b-service

echo "=== Pod Status ==="
kubectl get pod -l app=qwen3-8b -o wide

echo "=== Pod Logs (tail) ==="
kubectl logs -l app=qwen3-8b --tail=30

echo "=== Ingress Controller Status ==="
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx

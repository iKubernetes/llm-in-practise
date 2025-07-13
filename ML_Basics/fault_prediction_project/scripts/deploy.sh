#!/bin/bash
docker build -t fault-prediction-service:latest .
docker push fault-prediction-service:latest
kubectl apply -f kubernetes/fault_prediction_deployment.yaml
kubectl apply -f kubernetes/model_retrain_cronjob.yaml

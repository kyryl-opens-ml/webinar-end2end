kind create cluster --name workshop-end2end

kubectl create -f k8s/data.yaml
kubectl create -f k8s/experiments.yaml
kubectl create -f k8s/pipeline.yaml

kubectl create -f k8s/serving-custom-model.yaml
kubectl create -f k8s/serving-open-model.yaml

kubectl create -f k8s/monitoring-custom-model.yaml
kubectl create -f k8s/monitoring-open-model.yaml

# Webinar: "ML in Production" on practice

![alt text](./docs/end2end.jpg)

# Webinar and discord

- For a video lecture explaining this code, [watch this webinar](https://edu.kyrylai.com/courses/webinar-machine-learning-in-production)
- For support and questions, join this [Discord server](https://discord.gg/RNjfNrrN)

# Tools to install 

- [docker](https://docs.docker.com/engine/install/)
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [k9s](https://k9scli.io/topics/install/)

# Setup end2end 

```
bash ./script/build_docker.sh
bash ./script/start_all.sh
```

# Clean up

```
./script/start_all.sh
```

# Setup development

```
python -m venv ~/env/
source ~/env/bin/activate
pip install -r requirements.txt
```

# Observe 

```
k9s -A

kubectl port-forward --address 0.0.0.0 svc/data-labeling 6900:6900
kubectl port-forward --address 0.0.0.0 svc/monitoring-custom 8081:8080
kubectl port-forward --address 0.0.0.0 svc/monitoring-open 8082:8080
kubectl port-forward --address 0.0.0.0 svc/serving-custom-model 8001:80
kubectl port-forward --address 0.0.0.0 svc/serving-open-model 8002:80
```


# Setup creds 

```
export ARGILLA_URI=http://0.0.0.0:6900
export ARGILLA_KEY=adminadmin
export ARGILLA_NAMESPACE=admin
expoer HF_TOKEN=hf_your_token
```

# Data 

```
python end2end/data.py load-text-to-sql-dataset
python end2end/data.py load-data-for-labeling --dataset-name text2sql --sample --num-sample 10000
```

Reference 

- https://docs.argilla.io/en/latest/getting_started/quickstart_workflow_feedback.html
- https://github.com/argilla-io/argilla

# Experiments 


```
python end2end/experiments.py --model_name google/flan-t5-small --dataset_name text2sql-workshop --api_url ${ARGILLA_URI} --api_key ${ARGILLA_KEY} --workspace ${ARGILLA_NAMESPACE} --output_dir result-flan-t5-small --overwrite_output_dir --do_train --do_eval --evaluation_strategy steps --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 1e-3 --num_train_epochs 1000 --hub_model_id kyryl-opens-ml/flan-t5-small-sql --hub_token ${HF_TOKEN}
```

```
accelerate launch end2end/experiments.py --model_name google/flan-t5-small --dataset_name text2sql-workshop --api_url ${ARGILLA_URI} --api_key ${ARGILLA_KEY} --workspace ${ARGILLA_NAMESPACE} --output_dir result-flan-t5-small --overwrite_output_dir --do_train --do_eval --evaluation_strategy steps --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate 1e-3 --num_train_epochs 1000 --hub_model_id kyryl-opens-ml/flan-t5-small-sql --hub_token ${HF_TOKEN}
```


Reference

- https://arxiv.org/abs/2210.11416
- https://github.com/huggingface/peft
- https://huggingface.co/codellama/CodeLlama-7b-hf
- https://huggingface.co/google/gemma-7b
- https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2


# Pipeline

```
dagster dev -f end2end/pipeline.py -p 3000 -h 0.0.0.0
```

Reference

- https://dagster.io/blog/finetuning-llms


# Serving 

```
docker run --shm-size 1g -p 8080:80 ghcr.io/huggingface/text-generation-inference:1.4 --model-id kyryl-opens-ml/flan-t5-small-sql
```

Reference

- https://huggingface.co/docs/text-generation-inference/en/index
- https://github.com/predibase/lorax


# Monitoring

```
streamlit run --server.port 8080 --server.address 0.0.0.0 end2end/monitoring_ui.py
```

Reference

- https://docs.argilla.io/en/latest/getting_started/quickstart_workflow_feedback.html

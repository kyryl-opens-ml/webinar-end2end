# https://hub.docker.com/r/argilla/argilla-quickstart/tags
FROM argilla/argilla-quickstart:latest as data

ENV OWNER_USERNAME admin
ENV OWNER_PASSWORD adminadmin
ENV OWNER_API_KEY adminadmin

ENV ADMIN_USERNAME admin
ENV ADMIN_PASSWORD adminadmin
ENV ADMIN_API_KEY adminadmin

ENV ANNOTATOR_USERNAME admin 
ENV ANNOTATOR_PASSWORD adminadmin

ENV LOAD_DATASETS none 

COPY requirements-data.txt requirements-data.txt
RUN pip install -r requirements-data.txt

COPY . .


FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel as experiments
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    git-lfs

RUN pip install --upgrade pip
COPY requirements-experiments.txt requirements-experiments.txt
RUN pip install -r requirements-experiments.txt

COPY . .



FROM experiments as pipeline

WORKDIR /app


ENV DAGSTER_HOME /data/pipelines
RUN mkdir -p $DAGSTER_HOME

COPY requirements-pipeline.txt requirements-pipeline.txt
RUN pip install -r requirements-pipeline.txt

CMD touch $DAGSTER_HOME/dagster.yaml && dagster dev -p 3000 -h 0.0.0.0 -f /app/pipelines/main.py



FROM python:3.11.3 as monitoring

WORKDIR /app

COPY requirements-monitoring.txt requirements-monitoring.txt
RUN pip install -r requirements-monitoring.txt

COPY . .

CMD streamlit run --server.port 8080 --server.address 0.0.0.0 end2end/monitoring_ui.py
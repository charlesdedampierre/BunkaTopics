# How to
# copy .env.model to .env and write your own OPEN_AI_KEY
# docker build -t bunkatopicsapi .
# docker run --env-file .env -p 8000:8000 bunkatopicsapi
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3.9 \
    python3-pip \
    git \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# copying dependency
COPY bunkatopics /app/bunkatopics

# Rajouter variables environemment
ENV OPEN_AI_KEY=${OPEN_AI_KEY}

# Workspace
WORKDIR /app/api

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Python requirements (poetry has issues with fasttext: pybind11)
RUN pip install --upgrade pip

# - installing torch in a dedicated layer to ease image push to registry
#RUN pip install "torch==1.12.1" "torchvision==0.13.1"

# Other packages
#RUN pip install uvicorn
#RUN pip install sentence-transformers
#RUN pip install chromadb

# Bunka libraries
#COPY bunkatopics/requirements-bunka.txt requirements-bunka.txt
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
######################################
# changing user & downloading models #
######################################
FROM bunkatopicsbasedocker
# Models
RUN python -m spacy download en_core_web_sm
# reducing privilege
RUN groupadd rungroup && useradd -m -g rungroup runuser
RUN chown runuser:rungroup /app
USER runuser

# code
COPY --chown=runuser:rungroup api/run_server.sh run_server.sh
COPY --chown=runuser:rungroup api/bunka_api bunka_api
COPY --chown=runuser:rungroup api/celeryconfig.py celeryconfig.py

EXPOSE 8000

# start the server
CMD ["bash", "run_server.sh"]

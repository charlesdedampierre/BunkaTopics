# How to
# copy .env.model to .env and write your own OPEN_AI_KEY
# docker build -t bunkatopicsapi .
# docker run --env-file .env -p 8000:8000 bunkatopicsapi
FROM python:3.10 as bunkatopicsbasedocker

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-dev \
    python3 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# copying dependency
COPY bunkatopics /app/bunkatopics

# Rajouter variables environemment
ENV OPEN_AI_KEY=${OPEN_AI_KEY}

# Workspace
WORKDIR /app/api

# Python requirements (poetry has issues with fasttext: pybind11)
RUN pip install --upgrade pip

# Bunka libraries
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

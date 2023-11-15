# How to
# copy .env.model to .env and write your own OPEN_AI_KEY
# docker build -t bunkatopicsapi .
# docker run --env-file .env -p 8000:8000 bunkatopicsapi
FROM python:3.10 as bunkatopicsbasedocker

RUN apt update
RUN apt install -y python3-dev


# copying dependency
COPY bunkatopics /app/bunkatopics

# Workspace
WORKDIR /app/api

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
RUN pip install -r requirements.txt

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
COPY --chown=runuser:rungroup api/run_docker.sh run_docker.sh
COPY --chown=runuser:rungroup api/bunka_api bunka_api

EXPOSE 8000

# start the server
CMD ["bash", "run_docker.sh"]

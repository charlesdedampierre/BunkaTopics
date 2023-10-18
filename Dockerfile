FROM python:3.10

RUN apt update
RUN apt install -y python3-dev

# copying dependency
COPY bunkatopics /app/bunkatopics

# Workspace
WORKDIR /app/api

# Python requirements (poetry has issues with fasttext: pybind11)
RUN pip install --upgrade pip

# - installing torch in a dedicated layer to ease image push to registry
RUN pip install "torch==1.12.1" "torchvision==0.13.1"

RUN pip install uvicorn

# Models
RUN pip install spacy
RUN python -m spacy download en_core_web_sm

# Bunka libraries
COPY bunkatopics/requirements-bunka.txt requirements-bunka.txt
RUN pip install -r requirements-bunka.txt

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

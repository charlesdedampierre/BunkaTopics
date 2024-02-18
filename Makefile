SHELL := /bin/bash
.PHONY : all

default: 
	docker_build
	docker_build_worker
	docker_create_network

jupyter:
	python -m jupyterlab

test:
	python tests/test_bunka.py

test_fig:
	python tests/run_bunka.py

format:
	black bunkatopics
	isort bunkatopics

poetry_export:
	poetry shell
	poetry self add poetry-plugin-export
	poetry export --without-hashes --format=requirements.txt > requirements.txt

install_nginx_config:
	cp api/deployment/nginx-configuration-dev.conf /etc/nginx/sites-enabled/ && systemctl reload nginx

#############
# Streamlit  #
#############

run_streamlit:
	python -m streamlit run streamlit/app.py 

#############
# API  #
#############

run_api:
	python -m uvicorn api.bunka_api.routes:app

#############
# scaleway  #
#############

registry__login:
	docker login $(REGISTRY) -u nologin --password-stdin <<< $(SCW_SECRET_KEY)

registry__list:
	scw registry image list --profile $(PROFILE)

container__list:
	scw container container list --profile $(PROFILE)

container__get:
	scw container container get $(ID)

#############
# Docker API #
#############

docker_build:
	docker build -t $$API_IMAGE_NAME .

docker_run:
	docker run --restart=always --network bunkatopics_network --env-file .env -d --gpus all -p 8001:8000 --name $$API_CONTAINER_NAME $$API_IMAGE_NAME

docker_run_attach:
	docker run --network bunkatopics_network --env-file .env --gpus all -p 8001:8000 --name $$API_CONTAINER_NAME $$API_IMAGE_NAME

docker_tag:
	docker tag $$API_IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$API_IMAGE_NAME:latest

docker_push:
	docker push $$CONTAINER_REGISTRY_URL/$$API_IMAGE_NAME:latest


#############
# Docker CELERY WORKER #
#############

docker_create_network:
	docker network create bunkatopics_network

docker_build_worker:
	docker build -f DockerfileWorker -t $$WORKER_IMAGE_NAME .

docker_run_worker:
	docker run --restart=always --network bunkatopics_network --env-file .env -d --gpus all --name $$WORKER_CONTAINER_NAME $$WORKER_IMAGE_NAME

docker_run_worker_attach:
	docker run --network bunkatopics_network --env-file .env --gpus all --name $$WORKER_CONTAINER_NAME $$WORKER_IMAGE_NAME

docker_tag_worker:
	docker tag $$WORKER_IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$WORKER_IMAGE_NAME:latest

docker_push_worker:
	docker push $$CONTAINER_REGISTRY_URL/$$WORKER_IMAGE_NAME:latest

docker_run_redis:
	docker run --restart=always --network bunkatopics_network -d -p 6379:6379 --name redis redis


#docker run --restart=always --network bunkatopics_network --env-file .env -p 8001:8000 --name bunkaapi bunkatopicsapi
#docker run --restart=always --network bunkatopics_network --env-file .env -d --gpus all --name bunkaworker bunkatopicsworker
#docker build -f DockerfileWorker -t bunkatopicsworker .


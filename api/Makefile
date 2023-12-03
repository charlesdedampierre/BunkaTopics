SHELL := /bin/bash
.PHONY : all

help:
	cat Makefile

##########
# PROD   #
##########

run:
	python -m uvicorn bunka_api.routes:app

##########
# DEV    #
##########

format_code:
	python -m black bunka_api/*.py

run_dev: 
	python -m uvicorn bunka_api.routes:app --reload

run_worker: 
	python -m celery worker -l INFO

#############
# Docker CELERY WORKER #
#############

docker_build:
	docker build -t $$WORKER_IMAGE_NAME .

docker_run:
	docker run --restart=always --env-file .env -d --gpus all -p 6379:6379 --name $$WORKER_CONTAINER_NAME $$WORKER_IMAGE_NAME

docker_run_attach:
	docker run --env-file .env --gpus all -p 6379:6379 --name $$WORKER_CONTAINER_NAME $$WORKER_IMAGE_NAME

docker_tag:
	docker tag $$WORKER_IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$WORKER_IMAGE_NAME:latest

docker_push:
	docker push $$CONTAINER_REGISTRY_URL/$$WORKER_IMAGE_NAME:latest

docker_run_redis:
	 docker run --restart=always -d -p 6379:6379 --name redis redis
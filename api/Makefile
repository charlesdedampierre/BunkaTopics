SHELL := /bin/bash
.PHONY : all

help:
	cat Makefile

##########
# PROD   #
##########

run:
	python -m uvicorn src.bunka_api.main:app

##########
# DEV    #
##########


run_dev: format_code
	 python -m uvicorn src.bunka_api.main:app --reload

# CI/CD

poetry_export:
	poetry export --without-hashes --format=requirements.txt > requirements.txt

docker_build:
	docker build . -t bunka-api

docker_run:
	docker run --network host bunka-api:latest
	# docker run --network host --env-file .env bunka-api:latest

docker_build_run: docker_build docker_run

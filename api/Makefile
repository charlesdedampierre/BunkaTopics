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
SHELL := /bin/bash
.PHONY : all

help:
	cat Makefile

##########
# PROD   #
##########

run:
	python -m uvicorn bunka_api.main:app

##########
# DEV    #
##########

format_code:
	python -m black bunka_api/*.py

run_dev: 
	python -m uvicorn src.bunka_api.main:app --reload
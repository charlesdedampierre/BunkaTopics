SHELL := /bin/bash
.PHONY : all

#############
# DEV #
#############

# python3 -m venv bunka_env
# source bunka_env/bin/activate
# rm -r bunka_env (delete the environment)


install_packages:
	pip install -e .
	pip install -e '.['dev']'

docs_serve:
	mkdocs serve

### TO PUBLISH THIS WORKS AFTER MOVING THE POETRY FILE
# build_poetry:
# python -m build --sdist --wheel

# python setup.py sdist

pypi:
	python setup.py bdist_wheel --universal

pypi_publish:
	twine upload dist/* -u __token__ -p $(PYPY_TOKEN)
### TO PUBLISH THIS WORKS

# pypi_publish_test:
# 	twine upload --repository testpypi dist/* -u __token__ -p $(PYPY_TOKEN)

default: 
	docker_build
	docker_build_worker
	docker_create_network

jupyter:
	python -m jupyterlab

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

delete_checkpoints:
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +

format_code: clean
	black bunkatopics
	isort bunkatopics
	flake8 bunkatopics

test:
	python tests/test_bunka.py

check:
	python -m unittest tests/test_bunka.py

test_fig:
	python tests/run_bunka.py

poetry_export:
	poetry export --without-hashes --format=requirements.txt > requirements.txt

poetry_export_full:
	poetry shell
	poetry self add poetry-plugin-export
	poetry export --without-hashes --format=requirements.txt > requirements.txt	

pre_push: format_code clean check

#############
# DEV #
#############


install_nginx_config:
	cp api/deployment/nginx-configuration-dev.conf /etc/nginx/sites-enabled/ && systemctl reload nginx


tree_wihtout_pycache:
	tree bunkatopics -I '__pycache__'


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

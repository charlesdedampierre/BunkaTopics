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
	docker run --restart=always --env-file .env -d --gpus all -p 8000:8000 --name $$API_CONTAINER_NAME $$API_IMAGE_NAME

docker_run_attach:
	docker run --env-file .env --gpus all -p 8000:8000 --name $$API_CONTAINER_NAME $$API_IMAGE_NAME

docker_tag:
	docker tag $$API_IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$API_IMAGE_NAME:latest

docker_push:
	docker push $$CONTAINER_REGISTRY_URL/$$API_IMAGE_NAME:latest


#############
# Docker CELERY WORKER #
#############

run_worker: 
	python -m celery worker -l INFO

docker_build_worker:
	docker build -t $$WORKER_IMAGE_NAME .

docker_run_worker:
	docker run --restart=always --env-file ../.env -d --gpus all -p 6379:6379 --name $$WORKER_CONTAINER_NAME $$WORKER_IMAGE_NAME

docker_run_worker_attach:
	docker run --env-file ../.env --gpus all -p 6379:6379 --name $$WORKER_CONTAINER_NAME $$WORKER_IMAGE_NAME

docker_tag_worker:
	docker tag $$WORKER_IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$WORKER_IMAGE_NAME:latest

docker_push_worker:
	docker push $$CONTAINER_REGISTRY_URL/$$WORKER_IMAGE_NAME:latest

docker_run_redis:
	 docker run --restart=always -d -p 6379:6379 --name redis redis
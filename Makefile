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
	python -m uvicorn api.bunka_api.main:app

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
# Docker  #
#############

docker_build:
	docker build -t $$IMAGE_NAME

docker_run:
	docker run --env-file .env -d -p 8000:8000 $$IMAGE_NAME

docker_run_attach:
	docker run --env-file .env -p 8000:8000 $$IMAGE_NAME

docker_tag:
	docker tag $$IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$IMAGE_NAME:latest

docker_push:
	docker push $$CONTAINER_REGISTRY_URL/$$IMAGE_NAME:latest

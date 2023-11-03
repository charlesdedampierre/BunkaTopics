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
# Docker  #
#############


docker_build:
	docker build -t bunkatopics .

docker_run:
	docker run -p 8000:8000 bunkatopics

#############
# scaleway  #
#############

#REGISTRY = rg.fr-par.scw.cloud/bunkatopics
REGISTRY = rg.fr-par.scw.cloud/$(SCW_CONTAINER_REGISTRY_ID)
IMAGE_NAME = bunkatopics:latest

registry__login:
	docker login $(REGISTRY) -u nologin --password-stdin <<< $(SCW_SECRET_KEY)

registry__list:
	scw registry image list --profile $(PROFILE)

container__list:
	scw container container list --profile $(PROFILE)

container__get:
	scw container container get $(ID)

docker_tag:
	docker tag $(IMAGE_NAME) $(REGISTRY)/$(IMAGE_NAME)

docker_push:
	docker push $(REGISTRY)/$(IMAGE_NAME)

#REGISTRY = rg.fr-par.scw.cloud/bunkatopics
REGISTRY = rg.fr-par.scw.cloud/funcscwbunkatopicskrug38hz
IMAGE_NAME = bunkatopics:latest


jupyter:
	python -m jupyterlab

test:
	python tests/test_bunka.py

test_fig:
	python tests/run_bunka.py

run_streamlit:
	python -m streamlit run streamlit/app.py 

format:
	black bunkatopics
	isort bunkatopics

docker_build:
	docker build -t bunkatopics .

docker_run:
	docker run -p 8000:8000 bunkatopics

poetry_export:
	poetry export --without-hashes --format=requirements.txt > requirements.txt

#############
# registry  #
#############

registry__login:
	docker login $(REGISTRY) -u nologin --password-stdin <<< $(SCW_SECRET_KEY)

registry__list:
	scw registry image list --profile $(PROFILE)

#############
# scaleway  #
#############

container__list:
	scw container container list --profile $(PROFILE)

container__get:
	scw container container get $(ID)

docker_tag:
	docker tag $(IMAGE_NAME) $(REGISTRY)/$(IMAGE_NAME)

docker_push:
	docker push $(REGISTRY)/$(IMAGE_NAME)
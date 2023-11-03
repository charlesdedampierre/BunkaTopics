start:
	npm start

#############
# Docker  #
#############

docker_build:
	docker build -t bunkatopics_web .

docker_run:
	docker run -d -p 8080:80 bunkatopics_web

#############
# Registry  #
#############

REGISTRY = rg.fr-par.scw.cloud/$(SCW_CONTAINER_REGISTRY_ID)
IMAGE_NAME = bunkatopics_web:latest

docker_tag:
	docker tag $(IMAGE_NAME) $(REGISTRY)/$(IMAGE_NAME)

docker_push:
	docker push $(REGISTRY)/$(IMAGE_NAME)

registry__login:
	docker login $(REGISTRY) -u nologin --password-stdin <<< $(SCW_SECRET_KEY)

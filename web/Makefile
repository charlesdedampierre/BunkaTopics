IMAGE_NAME = bunkatopics_web

start:
	npm start

#############
# Docker  #
#############

docker_build:
	docker build -t $(IMAGE_NAME) .

docker_run:
	docker run -d -p 8080:80 $(IMAGE_NAME)

#############
# Registry  #
#############


docker_tag:
	echo "$(CONTAINER_REGISTRY_URL)"
	docker tag $(IMAGE_NAME) $(CONTAINER_REGISTRY_URL)/$(IMAGE_NAME):latest

docker_push:
	docker push $(CONTAINER_REGISTRY_URL)/$(IMAGE_NAME):latest

registry__login:
	docker login $(CONTAINER_REGISTRY_URL) -u nologin --password-stdin <<< $(SCW_SECRET_KEY)

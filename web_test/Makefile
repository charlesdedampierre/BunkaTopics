start:
	npm start

push:
	git add .
	git commit -m 'automatic commit4'
	git push


#############
# Docker  #
#############

docker_build:
	docker build -t bunkatopics .

docker_run:
	docker run -d -p 8080:80 bunkatopics

#############
# Registry  #
#############

REGISTRY = rg.fr-par.scw.cloud/bunka-docker-images
IMAGE_NAME = bunkatopics:latest

docker_tag:
	docker tag $(IMAGE_NAME) $(REGISTRY)/$(IMAGE_NAME)

docker_push:
	docker push $(REGISTRY)/$(IMAGE_NAME)

registry__login:
	docker login $(REGISTRY) -u nologin --password-stdin <<< $(SCW_SECRET_KEY)

registry__list:
	scw registry image list --profile $(PROFILE)

container__list:
	scw container container list --profile $(PROFILE)

container__get:
	scw container container get $(ID)
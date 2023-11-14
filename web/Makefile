docker_run:
	docker run -d -p 8080:80 $$IMAGE_NAME

docker_build:
	docker build -t $$IMAGE_NAME .

docker_tag:
	docker tag $$IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$IMAGE_NAME:latest

docker_push:
	docker push $$CONTAINER_REGISTRY_URL/$$IMAGE_NAME:latest

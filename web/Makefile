docker_run:
	docker run --env REACT_APP_API_ENDPOINT=$$REACT_APP_API_ENDPOINT --restart=always -d -p 8080:80 --name $$FRONT_CONTAINER_NAME $$FRONT_IMAGE_NAME

docker_build:
	docker build --build-arg REACT_APP_API_ENDPOINT=$$REACT_APP_API_ENDPOINT -t $$FRONT_IMAGE_NAME .

docker_tag:
	docker tag $$FRONT_IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$FRONT_IMAGE_NAME:latest

docker_push:
	docker push $$CONTAINER_REGISTRY_URL/$$FRONT_IMAGE_NAME:latest

#http://localhost:8080:80

registry__login:
	docker login $$CONTAINER_REGISTRY_URL -u nologin --password $$SCW_SECRET_KEY
docker_run:
	docker run -d -p 8080:80 $$IMAGE_NAME

docker_build:
	docker build --build-arg REACT_APP_API_ENDPOINT=/api -t $$IMAGE_NAME .

docker_tag:
	docker tag $$IMAGE_NAME $$CONTAINER_REGISTRY_URL/$$IMAGE_NAME:latest

docker_push:
	docker push $$CONTAINER_REGISTRY_URL/$$IMAGE_NAME:latest

#http://localhost:8080:80

registry__login:
	docker login $$CONTAINER_REGISTRY_URL -u nologin --password $$SCW_SECRET_KEY
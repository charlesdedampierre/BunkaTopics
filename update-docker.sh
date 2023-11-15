#!/bin/bash

# Define the containers and image names
FRONT_CONTAINER_NAME=bunkafront
API_CONTAINER_NAME=bunkaapi
FRONT_IMAGE_NAME=rg.fr-par.scw.cloud/funcscwbunkafront2bxzn2j5/bunkatopicsweb:latest 
API_IMAGE_NAME=rg.fr-par.scw.cloud/funcscwbunkafront2bxzn2j5/bunkatopicsapi:latest 

# Pull the latest version of the images
docker pull $FRONT_IMAGE_NAME
docker pull $API_IMAGE_NAME

# Check if the container is running and stop it
if [ "$(docker ps -q -f name=$API_CONTAINER_NAME)" ]; then
    docker stop $API_CONTAINER_NAME
    docker rm $API_CONTAINER_NAME
fi

# Start the container with the new image API_IMAGE_NAME
docker run -d -p 127.0.0.1:8080:80 --restart always --name $API_CONTAINER_NAME $API_IMAGE_NAME

# Check if the container is running and stop it
if [ "$(docker ps -q -f name=$FRONT_CONTAINER_NAME)" ]; then
    docker stop $FRONT_CONTAINER_NAME
    docker rm $FRONT_CONTAINER_NAME
fi

# Start the container with the new image FRONT_IMAGE_NAME
docker run -d -p 127.0.0.1:8080:80 --restart always --name $FRONT_CONTAINER_NAME $FRONT_IMAGE_NAME

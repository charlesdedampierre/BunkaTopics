#!/bin/bash

# Define your container and image names
CONTAINER_NAME=bunkafront
IMAGE_NAME=rg.fr-par.scw.cloud/funcscwbunkafront2bxzn2j5/bunkatopicsweb:latest 

# Pull the latest version of the image
docker pull $IMAGE_NAME

# Check if the container is running and stop it
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Start the container with the new image
docker run -d -p 127.0.0.1:8080:80 --restart always --name $CONTAINER_NAME $IMAGE_NAME

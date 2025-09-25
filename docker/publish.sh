#!/bin/bash

set -e

# Change this if your current user is not the same as your Docker Hub username
DOCKER_USER=$(whoami)
echo "Tagging image for user: $DOCKER_USER"
docker tag dllm-dev:py312-cuda124-torch251 $DOCKER_USER/dllm-dev:py312-cuda124-torch251
docker tag dllm-dev:py311-cuda128-torch270 $DOCKER_USER/dllm-dev:py311-cuda128-torch270

# Enter token at prompt; see:
# https://docs.docker.com/security/access-tokens/
echo "Logging into Docker Hub..."
docker login -u $DOCKER_USER

# Push image to Docker Hub under your username
echo "Pushing image to Docker Hub..."
docker push $DOCKER_USER/dllm-dev:py312-cuda124-torch251
docker push $DOCKER_USER/dllm-dev:py311-cuda128-torch270

echo "Publish complete!"
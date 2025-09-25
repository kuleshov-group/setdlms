#!/bin/bash

# Build script for DLLM-Dev Docker image;
# Run this script from the project root directory
set -e

echo "Building dllm-dev Docker images..."

# Build the Docker images
docker build --progress=plain \
  -f "docker/Dockerfile-py312-cuda124-torch251" \
  -t dllm-dev:py312-cuda124-torch251 .

docker build --progress=plain \
  -f "docker/Dockerfile-py311-cuda128-torch270" \
  -t dllm-dev:py311-cuda128-torch270 .


# Try them out
echo "Build complete"
echo "To run the containers:"
echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace dllm-dev:py312-cuda124-torch251"
echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace dllm-dev:py311-cuda128-torch270"

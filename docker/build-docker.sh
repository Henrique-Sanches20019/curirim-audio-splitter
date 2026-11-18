#!/bin/bash
# Build the training image "curirim". Run from anywhere: bash docker/build-docker.sh
set -e

# Build context = repository root (only pyproject.toml and uv.lock pass .dockerignore).
cd "$(dirname "$0")/.."
docker build -t curirim -f docker/Dockerfile .

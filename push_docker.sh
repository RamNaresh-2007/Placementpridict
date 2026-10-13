#!/bin/bash
set -e

IMAGE_NAME="ram032007naresh/placement-predict"
TAG="${1:-latest}"
FULL_TAG="${IMAGE_NAME}:${TAG}"

echo "=================================================="
echo " Docker Build & Push: ${FULL_TAG}"
echo "=================================================="

# Check docker CLI
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    exit 1
fi

# Ensure docker daemon is running
if ! docker info &> /dev/null; then
    echo "[INFO] Starting docker daemon service..."
    sudo service docker start || true
fi

echo ""
echo "[1/3] Logging in to Docker Hub as 'ram032007naresh'..."
docker login -u ram032007naresh

echo ""
echo "[2/3] Building Docker image '${FULL_TAG}'..."
docker build -t "${FULL_TAG}" .

echo ""
echo "[3/3] Pushing '${FULL_TAG}' to Docker Hub..."
docker push "${FULL_TAG}"

echo ""
echo "=================================================="
echo " SUCCESS! Image pushed to Docker Hub:"
echo " https://hub.docker.com/r/${IMAGE_NAME}"
echo "=================================================="

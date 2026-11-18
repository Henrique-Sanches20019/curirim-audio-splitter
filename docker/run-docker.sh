#!/bin/bash
# Start a GPU container from the "curirim" image with the repository mounted at /workspace.
#
#   bash docker/run-docker.sh                     # interactive shell
#   bash docker/run-docker.sh python -m curirim.train --run-dir runs/unet --resume
#
# Runs as your host user, so files written to runs/ and musdb18-wav/ stay yours.
# Run it inside tmux/screen: with --rm, the container stops when the SSH session dies
# (training resumes from runs/<name>/last.pt with --resume).
set -e

cd "$(dirname "$0")/.."

# --shm-size: DataLoader workers (num_workers: 4) pass batches through /dev/shm;
#             Docker's 64 MB default makes them crash.
# MUSDB_PATH: decoded WAV tree inside the mounted repository.
docker run --gpus all \
    -p 6006:6006 \
    -it \
    --rm \
    --shm-size=8g \
    -u "$(id -u):$(id -g)" \
    -v "${PWD}:/workspace" \
    -e MUSDB_PATH=/workspace/musdb18-wav \
    curirim \
    "${@:-bash}"

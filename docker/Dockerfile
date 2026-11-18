# GPU training image for the university server.
# Layout follows https://github.com/cmranieri/docker_gpu_ssh (Dockerfile + build/run scripts).
#
# Only the dependencies live in the image (installed from uv.lock into /opt/venv).
# The repository itself is mounted at /workspace by run-docker.sh, so code edits
# need no rebuild; rebuild only when pyproject.toml or uv.lock change.
#
# No CUDA base image is needed: the locked torch wheel ships its own CUDA runtime
# (nvidia-* wheels). The host only needs the NVIDIA driver + NVIDIA Container Toolkit.
FROM python:3.12-slim-bookworm

# uv binary, same version as the laptop.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

# ffmpeg: musdb/stempeg need it to decode the .stem.mp4 files (scripts/decode_musdb.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Venv outside /workspace, so the bind mount of the repo does not hide it.
# PYTHONPATH: code comes from the mounted repo, not from an installed copy.
# HOME: the container runs as the host uid (run-docker.sh), which has no home directory.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/src \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

# Exact versions from uv.lock; the project itself is not installed (see PYTHONPATH).
COPY pyproject.toml uv.lock /tmp/curirim/
RUN cd /tmp/curirim \
    && uv sync --frozen --no-install-project \
    && rm -rf /tmp/curirim /root/.cache/uv

# TensorBoard.
EXPOSE 6006

WORKDIR /workspace

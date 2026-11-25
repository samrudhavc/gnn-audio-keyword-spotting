
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Add the Python 3.11 package repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.11 and system dependencies
RUN apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        git \
        build-essential \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated Python environment.
# This bootstraps pip through the venv package.
RUN python3.11 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 with CUDA 12.1
RUN python -m pip install \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies without replacing the selected PyTorch version
COPY SW/requirements.txt /tmp/requirements.txt

RUN printf "torch==2.5.1\ntorchaudio==2.5.1\nsympy==1.13.1\n" \
    > /tmp/constraints.txt && \
    grep -vE '^(torch|torchaudio|torchvision|nvidia-|triton|cuda-|sympy)' \
    /tmp/requirements.txt > /tmp/req_clean.txt && \
    python -m pip install \
        -c /tmp/constraints.txt \
        -r /tmp/req_clean.txt

WORKDIR /workspace

CMD ["/bin/bash"]

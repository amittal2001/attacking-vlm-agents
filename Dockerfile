# # Start from your base image
# FROM windowsarena/winarena:latest

# # Install CUDA runtime and cuDNN
# # RUN apt-get update && \
# #     apt-get install -y --no-install-recommends wget gnupg && \
# #     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
# #     dpkg -i cuda-keyring_1.1-1_all.deb && \
# #     rm cuda-keyring_1.1-1_all.deb && \
# #     apt-get update && \
# #     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
# #         cuda-runtime-11-8 \
# #         libcudnn8 && \
# #     rm -rf /var/lib/apt/lists/*

# # Copy requirements first (for caching)
# COPY requirements.txt /tmp/requirements.txt

# # Install Python dependencies
# RUN pip install --no-cache-dir -r /tmp/requirements.txt

# # Install CUDA repo: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu
# # RUN apt-get update && \
# #     apt-get install -y --no-install-recommends wget gnupg && \
# #     apt-get download libcublas-12-0 && \
# #     mkdir /tmp/contents && \
# #     dpkg-deb -xv libcublas-12-0_*.deb /tmp/contents/ && \
# #     mv /tmp/contents/usr/local/cuda-12.0/targets/x86_64-linux/lib/* /usr/local/cuda/lib64/ && \
# #     rm -rf /tmp/contents libcublas-12-0_*.deb && \
# #     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Copy source code
# COPY src/win-arena-container/client /client

# # (Optional) Prepend CUDA libs to LD_LIBRARY_PATH so bitsandbytes can find them
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}


# Use NVIDIA official CUDA 12.2 runtime as base
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Ensure Python & pip are available
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy requirements first (for Docker caching)
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies (CUDA 12.2 compatible)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your source code
COPY src/win-arena-container/client /client

# Optional: help bitsandbytes & PyTorch find CUDA libs
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /client

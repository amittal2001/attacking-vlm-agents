# For runner image
#FROM continuumio/miniconda3:latest

# Copy environment file
#COPY environment.yml /tmp/environment.yml
#COPY requirements.txt /tmp/requirements.txt

# Create conda environment from environment.yml
#RUN conda env create -f /tmp/environment.yml

# Install Azure CLI
#RUN apt-get update && \
#    apt-get install -y curl apt-transport-https lsb-release gnupg && \
#    curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null && \
#    AZ_REPO=$(lsb_release -cs) && \
#    echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | tee /etc/apt/sources.list.d/azure-cli.list && \
#    apt-get update && \
#    apt-get install -y azure-cli && \
#    rm -rf /var/lib/apt/lists/*

# Force container to use winarena env python & pip
#ENV PATH="/opt/conda/envs/winarena/bin:$PATH"
#ENV CONDA_DEFAULT_ENV=winarena

# For azaure image
# Start from the WAA base image
FROM windowsarena/winarena:latest

# Copy your requirements
COPY requirements.txt /tmp/requirements.txt

# Install extra dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

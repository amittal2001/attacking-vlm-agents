FROM continuumio/miniconda3:latest

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml

SHELL ["conda", "run", "-n", "winarena", "/bin/bash", "-c"]


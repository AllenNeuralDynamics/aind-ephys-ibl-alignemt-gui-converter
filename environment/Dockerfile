# hash:sha256:0f4c46facc8ebb24e9656c34988d39ca26a4f3756bc65e37442083d3a6ad86d0
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    one-api==2.6.0 \
    spikeinterface[full]==0.99.1 \
    wavpack-numcodecs==0.1.5

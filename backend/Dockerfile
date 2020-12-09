FROM nvidia/cuda:11.1-devel-ubuntu20.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /root
ENV HOME /root

RUN apt-get update

ENV CUDNN_VERSION 8.0.5.39

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.1 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.1 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get install -y --no-install-recommends \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    wget \
    curl \
    htop \
    zip \
    unzip \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev && \
    python3 -V && \
    python3 -m pip --no-cache-dir install --upgrade pip setuptools && \
    ln -s $(which python3) /usr/local/bin/python

COPY ./requirements.txt ./requirements.txt

# Copies all the relevant files to the docker image
ADD ./src ./src
ADD ./data ./data

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

RUN  echo "/usr/local/cuda/compat" >> /etc/ld.so.conf.d/cuda-10-0.conf && \
    ldconfig

ENTRYPOINT ["python", "-u" "run.py"]

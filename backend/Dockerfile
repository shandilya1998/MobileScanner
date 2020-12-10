FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

RUN apt-get -y update

RUN apt-get install -y --no-install-recommends \
    git \
    tar \
    build-essential \
    software-properties-common \
    ca-certificates \
    wget \
    curl \
    htop \
    zip \
    unzip

RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz && \
    tar –xf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    ./configure ––enable–optimizations && \
    make install && \
    python3 ––version

COPY ./requirements.txt ./requirements.txt

# Copies all the relevant files to the docker image
ADD ./src ./src
ADD ./data ./data

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "-u" "run.py"]

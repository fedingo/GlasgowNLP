FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

MAINTAINER Fedingo

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    wget vim \
    software-properties-common \
    locales \
    graphviz \
    unzip 
    
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en

RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get install -y python3.8 \
    python3-pip \
    python-setuptools

RUN pip3 install --force-reinstall --upgrade pip
RUN pip install tensorflow-gpu==2.0.0-rc1 tensorflow_datasets
RUN pip install torch torchvision
RUN pip install jupyter jupyterlab tokenizer
RUN pip install torchtext pytorch-nlp nltk line_profiler runipy dotmap transformers matplotlib
RUN pip install transformers tokenizers==0.4
RUN pip install jsonlines tree-sitter anytree pyformlang autopep8
RUN pip install librosa lmdb ipympl sklearn numba==0.48.0

RUN apt-get install -y ffmpeg 

RUN apt-get install -y curl sudo dirmngr apt-transport-https lsb-release ca-certificates
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
RUN apt -y install nodejs

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
RUN jupyter lab build

# Pylucene setup
RUN apt-get update -y
RUN apt-get install -y default-jdk ant

WORKDIR /usr/lib/jvm/default-java/jre/lib
RUN ln -s ../../lib amd64

WORKDIR /usr/src/pylucene
RUN apt-get install curl -y
RUN curl http://apache.mirror.anlx.net/lucene/pylucene/pylucene-8.3.0-src.tar.gz \
     | tar -xz --strip-components=1
RUN cd jcc \
   && NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python3 setup.py install
RUN make all install JCC='python3 -m jcc' ANT=ant PYTHON=python NUM_FILES=8

WORKDIR ..
RUN rm -rf pylucene
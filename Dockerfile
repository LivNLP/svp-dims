FROM ubuntu:24.04

# Set timezone and install required packages
RUN apt-get update && \
    apt-get install -y \
    tzdata \
    git \
    vim \
    tmux \
    libsndfile-dev \
    apt-utils \
    software-properties-common \
    language-pack-ja-base \
    language-pack-ja 

ENV TZ Asia/Tokyo
ENV LANG ja_JP.UTF-8

# Python installation
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    pip \
    python3.10 \
    python3.10-distutil \
    && ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --upgrade pip setuptools 

# Set working directory
WORKDIR /work

# Copy source code and data
COPY src/ /work/src/

# Non-root user creation
RUN groupadd mygroup && \
    useradd -m -s /bin/bash -u 2000 -g mygroup myuser
RUN chown -R myuser /work

# Switch to non-root user
USER myuser

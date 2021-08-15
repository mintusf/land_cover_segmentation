FROM nvidia/cuda:11.4.0-base-ubuntu20.04
LABEL maintainer="fmintus <mintusf@gmail.com>"

# install requirements
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
    bzip2 \
    git \
    python3\
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    unzip \
	vim \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Create pipenv environment
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# Add user
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN passwd -d user
RUN usermod -aG sudo user
USER user

RUN ["/bin/bash"]
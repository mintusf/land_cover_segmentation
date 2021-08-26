# install requirements
sudo apt-get update -y \
  && sudo apt-get install -y --no-install-recommends \
    sudo \
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
  && sudo apt-get clean \
  && sudo rm -rf /var/lib/apt/lists/*

# Create pipenv environment
sudo pip install -r ./docker/requirements.txt
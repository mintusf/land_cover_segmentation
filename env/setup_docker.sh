#!/bin/bash
# You need to change followings:
REPOSITORY_PATH="/home/fmintus/land_cover_tracking"
DOCKERFILE_PATH="./env"
IMAGE_NAME="fmintus/study:latest"
CONTAINER_NAME="fmintus_study"
GPU='"device=8,9,10,11"'
MEMORY="24G"
DATA_PATH="/raid/fmintus/sentinel_land_cover/seg_data"

cd $REPOSITORY_PATH
echo "Creating docker image"
docker build --network host -t $IMAGE_NAME $DOCKERFILE_PATH --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
echo "Finished !"

echo "Creating docker container"
docker run --gpus $GPU \
           --net host \
           -v $REPOSITORY_PATH:/workspace \
           -v $DATA_PATH:/data \
           --shm-size=$MEMORY \
           --name $CONTAINER_NAME \
           -itd $IMAGE_NAME
echo "Finished !"

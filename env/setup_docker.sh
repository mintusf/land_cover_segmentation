#!/bin/bash
# You need to change followings:
REPOSITORY_PATH="/home/fmintus/land_cover_tracking"
DOCKERFILE_PATH="./docker"
IMAGE_NAME="fmintus/rtc_02_model_classification:latest"
CONTAINER_NAME="fmintus_rtc02_model_classification"
GPU='"device=0,1,2,3"'
MEMORY="12G"

cd $REPOSITORY_PATH
echo "Creating docker image"
docker build --network host -t $IMAGE_NAME DOCKERFILE_PATH --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
echo "Finished !"

echo "Creating docker container"
docker run --gpus $GPU \
           --net host \
           -v $REPOSITORY_PATH:/workspace \
           -v /raid/RTC_02/recieved_data:/received_data \
           -v /mnt/USB_HDD:/received_data2 \
           --shm-size=$MEMORY \
           --name $CONTAINER_NAME \
           -itd $IMAGE_NAME
echo "Finished !"

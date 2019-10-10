#!/bin/bash

IMAGE_NAME=lednet_docker:latest

SCRIPT_DIR=$(cd $(dirname $0); pwd)

docker run -it --rm \
  --runtime=nvidia \
  --volume="$SCRIPT_DIR/:/root/catkin_ws/src/lednet_docker/" \
  --net="host" \
  $IMAGE_NAME \
  bash -c "source /ros_entrypoint.sh $@"

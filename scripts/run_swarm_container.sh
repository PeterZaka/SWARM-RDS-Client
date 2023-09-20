#! /bin/bash

# This file runs the Docker container with required networking ports so that
# the SWARM system can function. This also creates a cache directory for
# offline validation of License Keys, which is mounted into the container
# at runtime.

# Check if they have added an input arguement for the container name
if [ -z "$1" ]
then
    echo "Please provide the name of the Docker image you wish to run."
    exit 1
fi

# Check if the image exists on their machine
if [ "$(docker images -q $1 2> /dev/null)" == "" ]
then
    echo "The Docker image $1 does not exist on your machine."
    exit 1
fi

# Check if they have added an input arguement for the ROS_IP address. If they haven't
# then set it to localhost
if [ -z "$2" ]
then
    echo "No ROS_IP address provided. Setting ROS_IP to localhost."
    ROS_IP_ADDRESS="localhost"
else
    ROS_IP_ADDRESS=$2
fi

if [ -d .cache ]
then
    echo "Directory .cache exists."
else
    echo "Creating .cache directory"
    mkdir .cache
fi

docker run -it --rm --gpus=all --runtime=nvidia --network=host -e "ROS_IP_ADDRESS=$ROS_IP_ADDRESS" -v $pwd/.cache:/home/airsim_user/SWARMCore/core/.cache $1
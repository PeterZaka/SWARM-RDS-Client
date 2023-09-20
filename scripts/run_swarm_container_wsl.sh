#! /bin/bash

# This file runs the Docker container with required networking ports so that
# the SWARM system can function. This also creates a cache directory for
# offline validation of License Keys, which is mounted into the container
# at runtime.

if [ -d ".cache" ]
 echo "Directory .cache exists."
else
    echo "Creating .cache directory"
    $_mkdir -p ".cache"
fi

docker run -it --rm --gpus=all --publish=5002:5002/tcp --publish=80:80/tcp --publish=8888:8888/tcp --publish=443:443/tcp -v $pwd/.cache:/home/airsim_user/SWARMCore/core/.cache swarm_home_v1.5.0
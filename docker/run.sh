#!/bin/bash
docker run \
-p 8869:8869 \
--gpus all \
-v $ISIC2024_PROJECT_ROOT:/app \
-v $HOME/clearml.conf:/root/clearml.conf \
--shm-size=5g \
--network="host" \
--shm-size=5g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--name isic2024 \
-it $1 /bin/bash

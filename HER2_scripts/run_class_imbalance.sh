#! /bin/bash

set -o errexit

#######################################################
# Build the Docker image
#######################################################

docker build -t inwosu/class_imbalance .

#######################################################
# Run docker command
#######################################################

dockerCommand="docker run -i -t --rm \
    -u $(id -u):$(id -g) \
    -v $(pwd):/step_1 \
    -v $(pwd)/../Data:/Data \
    inwosu/class_imbalance"

time $dockerCommand python3 scripts/run_class_imbalance.py

# $dockerCommand bash

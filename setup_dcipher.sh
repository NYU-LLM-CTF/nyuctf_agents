#!/usr/bin/env bash

BASE=$(dirname $0)
# Build main docker image
cd $BASE/docker/multiagent && docker build -t ctfenv:multiagent .

cd -
echo "Installing python package"
pip install --editable .

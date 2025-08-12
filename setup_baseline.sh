#!/usr/bin/env bash

networkname="ctfnet"
network_exists=$(docker network ls | grep $networkname)
# Create network
if [ -z "$network_exists" ]; then
    docker network create ctfnet
else
    echo "Network ${networkname} already exists, skip!"
fi

BASE=$(dirname $0)
# Build main docker image
cd $BASE/docker/baseline && docker build --build-arg HOST_UID=$(id -u) -t ctfenv .

# Download and unpack Ghidra
if [ ! -d ghidra_11.0.1_PUBLIC ]; then
    wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.0.1_build/ghidra_11.0.1_PUBLIC_20240130.zip
    unzip ghidra_11.0.1_PUBLIC_20240130.zip
    rm ghidra_11.0.1_PUBLIC_20240130.zip
fi

cd -
echo "Installing python dependencies"
pip install -r requirements.txt

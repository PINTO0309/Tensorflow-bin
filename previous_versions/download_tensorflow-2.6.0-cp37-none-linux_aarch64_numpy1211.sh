#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1XnV8W38HGW4PmENmfQA5-5NrQ84SsTqX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1XnV8W38HGW4PmENmfQA5-5NrQ84SsTqX" -o tensorflow-2.6.0-cp37-none-linux_aarch64.whl
echo Download finished.

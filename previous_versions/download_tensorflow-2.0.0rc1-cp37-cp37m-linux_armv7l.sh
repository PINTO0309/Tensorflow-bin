#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14K6zgil4Sd1ZHADZaJAbnYMePFzzCfhH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14K6zgil4Sd1ZHADZaJAbnYMePFzzCfhH" -o tensorflow-2.0.0rc1-cp37-cp37m-linux_armv7l.whl
echo Download finished.

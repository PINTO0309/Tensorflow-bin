#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1x2uDYob-zKP5optIkPtPx2PTqc4sKEGY" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1x2uDYob-zKP5optIkPtPx2PTqc4sKEGY" -o tensorflow-2.6.0-cp37-none-linux_aarch64.whl
echo Download finished.

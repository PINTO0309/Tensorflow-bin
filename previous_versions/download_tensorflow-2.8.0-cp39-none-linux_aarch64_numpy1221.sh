#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ZwCFOLO003qz4CKzXBtih_9ol21d22Br" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ZwCFOLO003qz4CKzXBtih_9ol21d22Br" -o tensorflow-2.8.0-cp39-none-linux_aarch64.whl
echo Download finished.

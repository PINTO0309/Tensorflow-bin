#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1EShQuBJKDC0blvm-y_-y5e9DN9k2Yy5a" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1EShQuBJKDC0blvm-y_-y5e9DN9k2Yy5a" -o tensorflow-2.6.0rc1-cp37-none-linux_aarch64.whl
echo Download finished.

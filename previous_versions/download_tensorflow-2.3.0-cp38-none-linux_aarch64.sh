#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1EbHTYyBPhpirEjt4gGBynPteK5He7ama" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1EbHTYyBPhpirEjt4gGBynPteK5He7ama" -o tensorflow-2.3.0-cp38-none-linux_aarch64.whl
echo Download finished.

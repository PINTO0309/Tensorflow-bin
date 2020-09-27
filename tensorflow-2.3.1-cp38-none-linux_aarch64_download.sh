#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1PIcr6hZYbZJpzssophDcmkm5ivSkebVK" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1PIcr6hZYbZJpzssophDcmkm5ivSkebVK" -o tensorflow-2.3.1-cp38-none-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11GVD2mlknhhrlNQignGZkdGhIsKxeOje" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11GVD2mlknhhrlNQignGZkdGhIsKxeOje" -o tensorflow-2.4.0-cp38-none-linux_aarch64.whl
echo Download finished.

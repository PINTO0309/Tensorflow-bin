#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1I68kXNICTrz7xCkzRQDUhZzq7eDeP01i" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1I68kXNICTrz7xCkzRQDUhZzq7eDeP01i" -o tensorflow-2.2.0-cp37-cp37m-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1rztvxvVqdhgC3o5OF7gGLSrE05zUk4jn" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1rztvxvVqdhgC3o5OF7gGLSrE05zUk4jn" -o tensorflow-2.5.0rc0-cp37-none-linux_aarch64.whl
echo Download finished.

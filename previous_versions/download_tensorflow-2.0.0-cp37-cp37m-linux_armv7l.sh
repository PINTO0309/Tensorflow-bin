#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1VsOBD3g0VujC-ZiaW7pSov194z6CcU7p" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1VsOBD3g0VujC-ZiaW7pSov194z6CcU7p" -o tensorflow-2.0.0-cp37-cp37m-linux_armv7l.whl
echo Download finished.

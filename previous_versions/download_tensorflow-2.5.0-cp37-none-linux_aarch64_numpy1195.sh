#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1o__C1dBp2-8D0k5Ggj6aUU2alG6J_u55" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1o__C1dBp2-8D0k5Ggj6aUU2alG6J_u55" -o tensorflow-2.5.0-cp37-none-linux_aarch64.whl
echo Download finished.

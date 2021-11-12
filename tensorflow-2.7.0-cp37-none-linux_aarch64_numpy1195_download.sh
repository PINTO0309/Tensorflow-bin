#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1qfRscjgaUg3HVEejgp9nHCktsASkS2Cl" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1qfRscjgaUg3HVEejgp9nHCktsASkS2Cl" -o tensorflow-2.7.0-cp37-none-linux_aarch64.whl
echo Download finished.

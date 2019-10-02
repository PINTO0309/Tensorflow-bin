#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1DgzRmgjn2EB-cANflk2jnj9_auiHDl7A" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1DgzRmgjn2EB-cANflk2jnj9_auiHDl7A" -o tensorflow-1.11.0-cp35-cp35m-linux_aarch64.whl
echo Download finished.

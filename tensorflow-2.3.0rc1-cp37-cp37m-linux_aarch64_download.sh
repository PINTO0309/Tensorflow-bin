#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=15nVB11d_PXseIECTqKCMtau-bgR3ATfb" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=15nVB11d_PXseIECTqKCMtau-bgR3ATfb" -o tensorflow-2.3.0rc1-cp37-cp37m-linux_aarch64.whl
echo Download finished.

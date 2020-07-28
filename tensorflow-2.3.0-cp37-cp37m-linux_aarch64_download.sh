#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1sBv_Fx9yz6GfW-4zuB4k66bI9EVOKDv7" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1sBv_Fx9yz6GfW-4zuB4k66bI9EVOKDv7" -o tensorflow-2.3.0-cp37-cp37m-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hVow21JbfMAGCyyTZaDYoEpUeXCnvWaI" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hVow21JbfMAGCyyTZaDYoEpUeXCnvWaI" -o tensorflow-2.1.0-cp37-cp37m-linux_aarch64.whl
echo Download finished.

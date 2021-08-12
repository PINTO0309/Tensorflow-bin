#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hzv5FwsI5K-yywxfZWN5XzJRA2VoXoE5" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hzv5FwsI5K-yywxfZWN5XzJRA2VoXoE5" -o tensorflow-2.6.0-cp39-none-linux_aarch64.whl
echo Download finished.

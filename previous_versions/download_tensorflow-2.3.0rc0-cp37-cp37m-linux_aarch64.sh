#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1naI8CwEc0PRksOwnE7TvUW4WyMa4bzeV" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1naI8CwEc0PRksOwnE7TvUW4WyMa4bzeV" -o tensorflow-2.3.0rc0-cp37-cp37m-linux_aarch64.whl
echo Download finished.

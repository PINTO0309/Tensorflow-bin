#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1yxyIE2Jvh1CQ6frQSDl99LBdsGHrYBQ8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1yxyIE2Jvh1CQ6frQSDl99LBdsGHrYBQ8" -o tensorflow-2.7.0-cp39-none-linux_aarch64.whl
echo Download finished.

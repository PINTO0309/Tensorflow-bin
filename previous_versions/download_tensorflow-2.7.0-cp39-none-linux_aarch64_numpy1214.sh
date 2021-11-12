#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1bbr5x_0bQ1e-yIe0F4F-47G859NzYhvz" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1bbr5x_0bQ1e-yIe0F4F-47G859NzYhvz" -o tensorflow-2.7.0-cp39-none-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=17MC6r2OXKyS2maNXcWVFjj5TG3coPE3X" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=17MC6r2OXKyS2maNXcWVFjj5TG3coPE3X" -o tensorflow-2.4.0-cp37-none-linux_aarch64.whl
echo Download finished.

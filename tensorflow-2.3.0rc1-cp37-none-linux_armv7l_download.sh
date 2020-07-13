#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1e1Aa9fuNUiY0SyDBxrfNukPIxZTEiiz9" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1e1Aa9fuNUiY0SyDBxrfNukPIxZTEiiz9" -o tensorflow-2.3.0rc1-cp37-none-linux_armv7l.whl
echo Download finished.

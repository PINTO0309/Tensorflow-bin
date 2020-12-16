#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1WNY8qfxNdEcTwVvBh_8Wi9XQ9L2ZKDkv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1WNY8qfxNdEcTwVvBh_8Wi9XQ9L2ZKDkv" -o tensorflow-2.4.0-cp37-none-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1HOhdBnvzowsg3ujfI42duba0IuLD3SN8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1HOhdBnvzowsg3ujfI42duba0IuLD3SN8" -o tensorflow-1.14.0-cp37-cp37m-linux_aarch64.whl
echo Download finished.

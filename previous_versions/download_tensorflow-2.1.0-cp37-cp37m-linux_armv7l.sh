#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1twAdG4G5G7PswjL55D-Ei2Mkh21JBA4M" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1twAdG4G5G7PswjL55D-Ei2Mkh21JBA4M" -o tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl
echo Download finished.

https://drive.google.com/open?id=
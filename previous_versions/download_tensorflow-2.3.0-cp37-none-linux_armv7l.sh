#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11iycpKFZc267gryq06sdV1zrz0St_Kox" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11iycpKFZc267gryq06sdV1zrz0St_Kox" -o tensorflow-2.3.0-cp37-none-linux_armv7l.whl
echo Download finished.

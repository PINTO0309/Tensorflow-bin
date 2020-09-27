#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1o-H38Wpl38Hk3uByNukBWId8VieVwGt0" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1o-H38Wpl38Hk3uByNukBWId8VieVwGt0" -o tensorflow-2.3.1-cp37-none-linux_armv7l.whl
echo Download finished.

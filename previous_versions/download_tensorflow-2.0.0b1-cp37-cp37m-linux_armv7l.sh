#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1vfcqXK3kJI1gm5h4G1cF5xQHisS5OO3C" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1vfcqXK3kJI1gm5h4G1cF5xQHisS5OO3C" -o tensorflow-2.0.0b1-cp37-cp37m-linux_armv7l.whl
echo Download finished.

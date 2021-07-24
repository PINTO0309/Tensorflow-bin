#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1iM4vD2lpuaARCv4zvEzheI0XABiys6Nq" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1iM4vD2lpuaARCv4zvEzheI0XABiys6Nq" -o tensorflow-2.6.0rc1-cp38-none-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1D9IT2WC2mrWOixsNKohxiJJIlY2IzmQQ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1D9IT2WC2mrWOixsNKohxiJJIlY2IzmQQ" -o tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl
echo Download finished.

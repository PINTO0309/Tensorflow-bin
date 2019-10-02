#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ViufrUjvy2SNfeYy3dl6e4ynvr7DZPLy" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ViufrUjvy2SNfeYy3dl6e4ynvr7DZPLy" -o tensorflow-2.0.0b1-cp37-cp37m-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1iqylkLsgwHxB_nyZ1H4UmCY3Gy47qlOS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1iqylkLsgwHxB_nyZ1H4UmCY3Gy47qlOS" -o tensorflow-2.5.0-cp37-none-linux_armv7l.whl
echo Download finished.

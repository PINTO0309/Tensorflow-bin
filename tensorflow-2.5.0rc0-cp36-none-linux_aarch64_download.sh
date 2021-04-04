#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Dv1dMkMY18NIjY71mzDHjKuM7Z8mcbLf" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Dv1dMkMY18NIjY71mzDHjKuM7Z8mcbLf" -o tensorflow-2.5.0rc0-cp36-none-linux_aarch64.whl
echo Download finished.

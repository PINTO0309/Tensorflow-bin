#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1yL0WOKA339ETUdCbjkK7O3cLYJneo1F5" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1yL0WOKA339ETUdCbjkK7O3cLYJneo1F5" -o tensorflow-2.5.0-cp38-none-linux_aarch64.whl
echo Download finished.

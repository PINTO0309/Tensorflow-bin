#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11PQxiGqGszPKtWAWq8PXakyYyJU2ZARe" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11PQxiGqGszPKtWAWq8PXakyYyJU2ZARe" -o tensorflow-2.4.0-cp37-none-linux_armv7l.whl
echo Download finished.

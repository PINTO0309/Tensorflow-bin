#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1qfAKqNfWWHX0t017twOUAYTzdFa9k3Em" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1qfAKqNfWWHX0t017twOUAYTzdFa9k3Em" -o tensorflow-2.5.0-cp37-none-linux_armv7l.whl
echo Download finished.

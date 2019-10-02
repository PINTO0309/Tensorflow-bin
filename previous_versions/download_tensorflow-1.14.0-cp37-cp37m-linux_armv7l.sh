#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mUP6t4o7xBXhXDTTkS91uluZstqcnQj5" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mUP6t4o7xBXhXDTTkS91uluZstqcnQj5" -o tensorflow-1.14.0-cp37-cp37m-linux_armv7l.whl
echo Download finished.

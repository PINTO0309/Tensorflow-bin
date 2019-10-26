#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1jfc92gmDyt-sehIb9ygnzOSVDj7sy9oJ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1jfc92gmDyt-sehIb9ygnzOSVDj7sy9oJ" -o tensorflow-1.15.0-cp27-cp27mu-linux_armv7l.whl
echo Download finished.

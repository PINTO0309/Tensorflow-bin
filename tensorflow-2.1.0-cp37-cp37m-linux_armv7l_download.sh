#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1nAIYS-f-p-gmj1MzgZ9R3imtmrvlkna0" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1nAIYS-f-p-gmj1MzgZ9R3imtmrvlkna0" -o tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl
echo Download finished.

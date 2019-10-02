#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xY1sMi8yHzxcZ-RMwSzCeUJpNfQF-8jP" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xY1sMi8yHzxcZ-RMwSzCeUJpNfQF-8jP" -o tensorflow-2.0.0rc0-cp37-cp37m-linux_armv7l.whl
echo Download finished.
https://drive.google.com/open?id=
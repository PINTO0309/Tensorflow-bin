#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1dKQCz4CA0rz2utt0GmXEQWnIeQ4SxHO5" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1dKQCz4CA0rz2utt0GmXEQWnIeQ4SxHO5" -o tensorflow-2.4.0-cp37-none-linux_armv7l.whl
echo Download finished.

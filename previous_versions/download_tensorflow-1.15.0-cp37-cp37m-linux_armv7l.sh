#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LLtlgZxjDbh10v03WIJABxS_Q4tCOp3U" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LLtlgZxjDbh10v03WIJABxS_Q4tCOp3U" -o tensorflow-1.15.0-cp37-cp37m-linux_armv7l.whl
echo Download finished.

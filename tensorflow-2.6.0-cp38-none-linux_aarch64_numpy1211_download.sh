#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=12gqVbQvCcTG0OFnJtMVBmqstzgCAaLj9" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=12gqVbQvCcTG0OFnJtMVBmqstzgCAaLj9" -o tensorflow-2.6.0-cp38-none-linux_aarch64.whl
echo Download finished.

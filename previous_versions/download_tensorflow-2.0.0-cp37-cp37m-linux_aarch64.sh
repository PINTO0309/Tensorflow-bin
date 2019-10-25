#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1M7vQ4dV73SUcnoonyF6qjeYhLIYBTnGv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1M7vQ4dV73SUcnoonyF6qjeYhLIYBTnGv" -o tensorflow-2.0.0-cp37-cp37m-linux_aarch64.whl
echo Download finished.

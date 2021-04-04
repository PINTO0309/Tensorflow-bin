#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=10rNYmGu2V9PFCcSRUEp_6H8j9WeKPusR" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=10rNYmGu2V9PFCcSRUEp_6H8j9WeKPusR" -o tensorflow-2.5.0rc0-cp37-none-linux_armv7l.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1fw8n8dI-NkC5ZfKFhgB7LFTdkOqcgIfj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1fw8n8dI-NkC5ZfKFhgB7LFTdkOqcgIfj" -o tensorflow-2.3.1-cp37-cp37m-linux_aarch64.whl
echo Download finished.

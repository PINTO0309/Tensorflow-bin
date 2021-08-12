#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1kVXgUttQ-BoC9RplRFborVmqmmpGly0j" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1kVXgUttQ-BoC9RplRFborVmqmmpGly0j" -o tensorflow-2.6.0-cp39-none-linux_aarch64.whl
echo Download finished.

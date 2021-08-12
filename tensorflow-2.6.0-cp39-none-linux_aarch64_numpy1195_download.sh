#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1yK43Su_SdrCjmaKf7YyAv1Wudj7ddwdG" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1yK43Su_SdrCjmaKf7YyAv1Wudj7ddwdG" -o tensorflow-2.6.0-cp39-none-linux_aarch64.whl
echo Download finished.

#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1EsObTazsUxmIBj-37L3I2hTdXvVItQD8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1EsObTazsUxmIBj-37L3I2hTdXvVItQD8" -o tensorflow-2.5.0-cp39-none-linux_aarch64.whl
echo Download finished.

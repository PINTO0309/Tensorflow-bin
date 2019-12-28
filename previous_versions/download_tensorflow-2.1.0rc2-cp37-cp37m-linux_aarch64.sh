#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LwQEsyUUiRufxlta0Pci3_5aHehjNlmW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LwQEsyUUiRufxlta0Pci3_5aHehjNlmW" -o tensorflow-2.1.0rc2-cp37-cp37m-linux_aarch64.whl
echo Download finished.

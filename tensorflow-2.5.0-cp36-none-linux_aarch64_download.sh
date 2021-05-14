#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=184pwVCxRToGusUfwTNT5uNUTPtZyEvUW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=184pwVCxRToGusUfwTNT5uNUTPtZyEvUW" -o tensorflow-2.5.0-cp36-none-linux_aarch64.whl
echo Download finished.

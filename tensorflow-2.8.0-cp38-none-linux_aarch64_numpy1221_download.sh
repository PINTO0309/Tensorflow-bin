#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1upJD-J4Z4jPu5wTx3FW_KpK7t7oAj5Ac" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1upJD-J4Z4jPu5wTx3FW_KpK7t7oAj5Ac" -o tensorflow-2.8.0-cp38-none-linux_aarch64.whl
echo Download finished.

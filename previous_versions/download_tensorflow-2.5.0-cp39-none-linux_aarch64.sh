#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1fqpjsR2Jyso035gTDQ2yrLB2mMUSaX8L" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1fqpjsR2Jyso035gTDQ2yrLB2mMUSaX8L" -o tensorflow-2.5.0-cp39-none-linux_aarch64.whl
echo Download finished.

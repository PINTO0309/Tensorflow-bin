#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13jEjusLmfaI0yph1LHfMF6hbocNuZu2c" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13jEjusLmfaI0yph1LHfMF6hbocNuZu2c" -o tensorflow-2.0.0-cp27-cp27mu-linux_armv7l.whl
echo Download finished.

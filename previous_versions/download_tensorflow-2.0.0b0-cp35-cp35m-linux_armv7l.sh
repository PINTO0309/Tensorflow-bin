#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1M1cNfBWUFVPMZsB9M7kx5HsLlgqWLkan" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1M1cNfBWUFVPMZsB9M7kx5HsLlgqWLkan" -o tensorflow-2.0.0b0-cp35-cp35m-linux_armv7l.whl
echo Download finished.

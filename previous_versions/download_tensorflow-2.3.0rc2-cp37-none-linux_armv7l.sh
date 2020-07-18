#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Mdfx5bNr1fZGAB-oPWCChW503xCwz9pw" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Mdfx5bNr1fZGAB-oPWCChW503xCwz9pw" -o tensorflow-2.3.0rc2-cp37-none-linux_armv7l.whl
echo Download finished.

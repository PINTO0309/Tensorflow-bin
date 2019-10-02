#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Z23HAQX_1F1RRnVNynE08AVkrw8C4eRy" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Z23HAQX_1F1RRnVNynE08AVkrw8C4eRy" -o tensorflow-2.0.0rc2-cp37-cp37m-linux_armv7l.whl
echo Download finished.

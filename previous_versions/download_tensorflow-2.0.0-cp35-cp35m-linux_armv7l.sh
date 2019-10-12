#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1P5WYMZVKNIhae_WIGJWUER-3mr4Rk49u" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1P5WYMZVKNIhae_WIGJWUER-3mr4Rk49u" -o tensorflow-2.0.0-cp35-cp35m-linux_armv7l.whl
echo Download finished.

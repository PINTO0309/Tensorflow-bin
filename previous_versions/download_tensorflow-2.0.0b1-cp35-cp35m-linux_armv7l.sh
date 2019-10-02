#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mVxt3EaMX1hagCJMv43R8uzIFpZPkiHU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mVxt3EaMX1hagCJMv43R8uzIFpZPkiHU" -o tensorflow-2.0.0b1-cp35-cp35m-linux_armv7l.whl
echo Download finished.

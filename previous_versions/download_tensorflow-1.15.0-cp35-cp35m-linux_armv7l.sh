#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=17om3AW_gW8OYUJDwSceNuo8t-0G1_q1f" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=17om3AW_gW8OYUJDwSceNuo8t-0G1_q1f" -o tensorflow-1.15.0-cp35-cp35m-linux_armv7l.whl
echo Download finished.

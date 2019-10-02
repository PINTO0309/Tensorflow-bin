#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1bwxyzocEyjlM-9Ym0_YHZ6DzNrf7p0yu" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1bwxyzocEyjlM-9Ym0_YHZ6DzNrf7p0yu" -o tensorflow-2.0.0a0-cp35-cp35m-linux_armv7l.whl
echo Download finished.

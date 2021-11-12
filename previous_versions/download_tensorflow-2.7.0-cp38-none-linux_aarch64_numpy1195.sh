#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ArH-5LsHrE30jGgsD8jx4n7yLqEZLUdi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ArH-5LsHrE30jGgsD8jx4n7yLqEZLUdi" -o tensorflow-2.7.0-cp38-none-linux_aarch64.whl
echo Download finished.

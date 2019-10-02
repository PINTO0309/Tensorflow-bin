#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ZQZb-doMebMDAUZbEjNbZxP-YeOCJPQw" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ZQZb-doMebMDAUZbEjNbZxP-YeOCJPQw" -o tensorflow-1.14.0-cp35-cp35m-linux_armv7l.whl
echo Download finished.

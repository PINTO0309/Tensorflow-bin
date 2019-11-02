#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1XQs6CD1tiDmuA9MP_5MuN7lExaGAsweO" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1XQs6CD1tiDmuA9MP_5MuN7lExaGAsweO" -o packages.tar.gz
tar -zxvf packages.tar.gz
rm packages.tar.gz
echo Download finished.

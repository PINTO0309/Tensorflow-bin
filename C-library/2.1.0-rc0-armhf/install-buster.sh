#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1TDUlJ2Z6YNHY8P6FYeo284DXl3x9AY10" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1TDUlJ2Z6YNHY8P6FYeo284DXl3x9AY10" -o libtensorflow.tar.gz
tar -C /usr/local -xzf libtensorflow.tar.gz
rm libtensorflow.tar.gz
sudo ldconfig
echo 'Finish!!'

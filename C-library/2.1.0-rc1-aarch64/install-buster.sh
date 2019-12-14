#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=15MweNfBF9w7U2tObTuiw1iGthZKQgnf4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=15MweNfBF9w7U2tObTuiw1iGthZKQgnf4" -o libtensorflow.tar.gz
tar -C /usr/local -xzf libtensorflow.tar.gz
rm libtensorflow.tar.gz
sudo ldconfig
echo 'Finish!!'

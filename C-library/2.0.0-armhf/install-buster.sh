#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1_PY_oFkUt7DssXw1TLFF3Mpid6b-xcVQ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1_PY_oFkUt7DssXw1TLFF3Mpid6b-xcVQ" -o libtensorflow.tar.gz
tar -C /usr/local -xzf libtensorflow.tar.gz
rm libtensorflow.tar.gz
sudo ldconfig
echo 'Finish!!'

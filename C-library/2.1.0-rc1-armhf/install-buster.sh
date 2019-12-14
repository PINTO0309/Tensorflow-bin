#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1OeTlbB5ns-sVmxoa5-90R8J5qcwfZKSc" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1OeTlbB5ns-sVmxoa5-90R8J5qcwfZKSc" -o libtensorflow.tar.gz
tar -C /usr/local -xzf libtensorflow.tar.gz
rm libtensorflow.tar.gz
sudo ldconfig
echo 'Finish!!'

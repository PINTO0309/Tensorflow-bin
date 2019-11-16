#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Rrl_V6cm73GeIA9-zfyZ7BLQgnCrxz25" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Rrl_V6cm73GeIA9-zfyZ7BLQgnCrxz25" -o libtensorflow.tar.gz
tar -xzf libtensorflow.tar.gz
rm libtensorflow.tar.gz
sudo chmod -R 777 libtensorflow
echo 'Finish!!'


#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1v8Xn6OnBne3e_pqAwWn1nJiLqaKf_ehv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1v8Xn6OnBne3e_pqAwWn1nJiLqaKf_ehv" -o tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
echo Download finished.

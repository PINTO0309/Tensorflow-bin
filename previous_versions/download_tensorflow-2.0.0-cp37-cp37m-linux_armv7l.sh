#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1AUlqf3oosa6FLlkQgO-NSO1Ur2YutG9o" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1AUlqf3oosa6FLlkQgO-NSO1Ur2YutG9o" -o tensorflow-2.0.0-cp37-cp37m-linux_armv7l.whl
echo Download finished.

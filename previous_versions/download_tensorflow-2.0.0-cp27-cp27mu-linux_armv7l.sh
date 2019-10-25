#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1C5z-yr4Cf64_CvyRKUM31jhD89T_LA0E" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1C5z-yr4Cf64_CvyRKUM31jhD89T_LA0E" -o tensorflow-2.0.0-cp27-cp27mu-linux_armv7l.whl
echo Download finished.

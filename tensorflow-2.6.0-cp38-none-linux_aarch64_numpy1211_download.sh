#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hz1z0t_0kgMdInVElQJuE4NiqJi7IsZ_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hz1z0t_0kgMdInVElQJuE4NiqJi7IsZ_" -o tensorflow-2.6.0-cp38-none-linux_aarch64.whl
echo Download finished.

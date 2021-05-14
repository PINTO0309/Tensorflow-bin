#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1dLS8Faq0wuipbfBuzNgTl4Uv9jBj8K7c" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1dLS8Faq0wuipbfBuzNgTl4Uv9jBj8K7c" -o tensorflow-2.5.0-cp38-none-linux_aarch64.whl
echo Download finished.

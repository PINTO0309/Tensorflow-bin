#!/bin/bash

TFVER=2.12.0rc0
curl -L https://github.com/PINTO0309/Tensorflow-bin/releases/download/v${TFVER}/tensorflow-${TFVER}-cp39-none-linux_aarch64.whl -o tensorflow-${TFVER}-cp39-none-linux_aarch64.whl
echo Download finished.

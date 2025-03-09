#!/usr/bin/env bash

echo "if install fails due to python being not found, edit setup_no_captions.sh to replace with another version of python"

# if you are on a mac, you can try to replace "python3.10" with:
# python3.10
# python3.11 (not tested)
# python3.12 (not tested)
# python3.13 (tested, fails to install)

python3.10 -m venv .venv

source .venv/bin/activate

python3.10 -m pip install -r requirements_without_flash_attention.txt

# if you require flash attention, please install it manually for your operating system

# you can try this:
# python -m pip install wheel setuptools flash-attn --no-build-isolation --no-cache-dir
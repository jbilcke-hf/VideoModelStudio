#!/usr/bin/env bash

python -m venv .venv

source .venv/bin/activate

python -m pip install -r requirements.txt

# if flash attention couldn't be installed for your operating system you can try this:
# python -m pip install wheel setuptools flash-attn --no-build-isolation --no-cache-dir
#!/usr/bin/env bash

echo "Degraded setup for environments where decord cannot be installed"

# if you are on a mac, you can try to replace "python3.10" with:
# python3.10
# python3.11 (not tested)
# python3.12 (not tested)
# python3.13 (tested, fails to install)

python3.10 -m venv .venv

source .venv/bin/activate

# :(
python3.10 -m pip install -r degraded_requirements.txt

# :((
finetrainers @ git+https://github.com/a-r-r-o-w/finetrainers.git@main --no-deps

# :(((
python3.10 -m pip install -r degraded_finetrainers_requirements.txt

# if you require flash attention, please install it manually for your operating system

# you can try this:
# python -m pip install wheel setuptools flash-attn --no-build-isolation --no-cache-dir
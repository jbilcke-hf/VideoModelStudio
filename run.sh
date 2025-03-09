#!/usr/bin/env bash

source .venv/bin/activate

echo "if run.sh fails due to python being not found, edit run.sh to replace with another version of python"

# if you are on a mac, you can try to replace "python3.10" with:
# python3.10
# python3.11 (not tested)
# python3.12 (not tested)
# python3.13 (tested, fails to install)

USE_MOCK_CAPTIONING_MODEL=True python3.10 app.py
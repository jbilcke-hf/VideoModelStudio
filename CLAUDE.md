# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Video Model Studio - Guidelines for Claude

## Build & Run Commands
- Setup: `./setup.sh` (with flash attention) or `./degraded_setup.sh` (without flash-attention)
- Run: `./run.sh` or `python3.10 app.py` 
- Test: 
  - Full test: `python3 tests/test_dataset.py`
  - Single model test: `bash tests/scripts/dummy_cogvideox_lora.sh` (or other model variants)
- Run test suite: `bash tests/test_model_runs_minimally_lora.sh`

## Code Style
- Python version: 3.10 (required for flash-attention compatibility)
- Type hints: Use typing module annotations for all functions (from typing import Any, Optional, Dict, List, Union, Tuple)
- Docstrings: Google style with Args/Returns sections for all functions
- Error handling: Use try/except with specific exceptions, log errors appropriately
- Imports: Group standard lib, third-party, and project imports
- Naming: snake_case for functions/variables, PascalCase for classes
- Use Path objects from pathlib instead of string paths
- Extract reusable logic to separate utility functions
- Environment variables: Use parse_bool_env for boolean env vars
- Logging: Use the logging module with appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- UI components: Organize in tabs and use consistent naming for components dict
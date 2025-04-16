# Video Model Studio - Guidelines for Claude

## Build & Run Commands
- Setup: `./setup.sh` (with flash attention) or `./setup_no_captions.sh` (without)
- Run: `./run.sh` or `python3.10 app.py`
- Test: `python3 tests/test_dataset.py`
- Single model test: `bash tests/scripts/dummy_cogvideox_lora.sh`

## Code Style
- Python version: 3.10 (required for flash-attention compatibility)
- Type hints: Use typing module annotations for all functions
- Docstrings: Google style with Args/Returns sections
- Error handling: Use try/except with specific exceptions, log errors
- Imports: Group standard lib, third-party, and project imports
- Naming: snake_case for functions/variables, PascalCase for classes
- Use Path objects from pathlib instead of string paths
- Format utility functions: Extract reusable logic to separate functions
- Environment variables: Use parse_bool_env for boolean env vars
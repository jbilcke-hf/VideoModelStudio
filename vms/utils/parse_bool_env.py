from typing import Any, Optional, Dict, List, Union, Tuple

def parse_bool_env(env_value: Optional[str]) -> bool:
    """Parse environment variable string to boolean
    
    Handles various true/false string representations:
    - True: "true", "True", "TRUE", "1", etc
    - False: "false", "False", "FALSE", "0", "", None
    """
    if not env_value:
        return False
    return str(env_value).lower() in ('true', '1', 't', 'y', 'yes')

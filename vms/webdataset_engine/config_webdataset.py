"""
WebDataset configuration for Video Model Studio
"""

import os
from pathlib import Path
from typing import Dict, List, Union

# Shard naming conventions
SHARD_PATTERN = "{prefix}-{number:06d}.tar"

# Shard size settings
MAX_SHARD_SIZE_MB = 1024  # 1GB per shard
TARGET_SAMPLES_PER_SHARD = 1000  # Target number of samples per shard

# Metadata structure within shards
METADATA_FILENAME = "_metadata.json"

# Directory structure for shards
SHARD_DIRS = {
    "raw": "raw_shards",      # Initial import with minimal processing
    "processed": "processed_shards",  # After preprocessing (cropping, etc.)
    "training": "training_shards",  # Final training-ready shards (with captions)
}

# Extensions for different content types
EXTENSIONS = {
    "video": [".mp4", ".webm"],
    "image": [".jpg", ".jpeg", ".png", ".webp", ".avif"],
    "caption": [".txt", ".caption", ".json", ".cls"],
}

# Streaming settings
BUFFER_SIZE = 1000  # Number of samples to buffer for shuffling
PREFETCH_FACTOR = 2  # How many batches to prefetch

# Training cache settings
CACHE_ENABLED = True
MAX_CACHE_SIZE_GB = 10
CACHE_EXPIRATION_DAYS = 7

def get_shard_path(directory: Union[str, Path], prefix: str, number: int) -> Path:
    """Generate shard path with proper naming convention"""
    return Path(directory) / SHARD_PATTERN.format(prefix=prefix, number=number)

def get_all_shards(directory: Union[str, Path], prefix: str) -> List[Path]:
    """Get all shards in a directory with given prefix"""
    pattern = f"{prefix}-*.tar"
    return sorted(Path(directory).glob(pattern))
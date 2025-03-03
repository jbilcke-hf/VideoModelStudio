"""
WebDataset module initialization.
This ensures all WebDataset-related modules are properly imported and registered.
"""

from pathlib import Path
import logging

from .config_webdataset import (
    SHARD_DIRS, SHARD_PATTERN, MAX_SHARD_SIZE_MB, 
    TARGET_SAMPLES_PER_SHARD, METADATA_FILENAME,
    EXTENSIONS, BUFFER_SIZE, PREFETCH_FACTOR,
    CACHE_ENABLED, MAX_CACHE_SIZE_GB, CACHE_EXPIRATION_DAYS,
    get_shard_path, get_all_shards
)

from .webdataset_manager import WebDatasetManager
from .shard_writer import ShardWriter
from .webdata_captioner import WebDatasetCaptioningService
from .webdataset_import_service import WebDatasetImportService
from .webdataset_processing_service import WebDatasetProcessingService
from .webdataset_processor import WebDatasetProcessor

from ..config import STORAGE_PATH

logger = logging.getLogger(__name__)

# Create shard directories on module import
def setup_webdataset_directories():
    """Ensure all required WebDataset directories exist"""
    try:
        storage_path = STORAGE_PATH
        storage_path.mkdir(parents=True, exist_ok=True)
        
        for dirname in SHARD_DIRS.values():
            shard_dir = storage_path / dirname
            shard_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"WebDataset directories initialized at {storage_path}")
        
    except Exception as e:
        logger.error(f"Error setting up WebDataset directories: {e}")

# Run setup when module is imported
setup_webdataset_directories()
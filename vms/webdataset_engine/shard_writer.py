"""
WebDataset shard writer for Video Model Studio
"""

import os
import json
import tarfile
import tempfile
import logging
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any, Set
from datetime import datetime
import io

from .config_webdataset import (
    SHARD_PATTERN, MAX_SHARD_SIZE_MB, TARGET_SAMPLES_PER_SHARD,
    METADATA_FILENAME, SHARD_DIRS, EXTENSIONS
)

logger = logging.getLogger(__name__)

class ShardWriter:
    """Writer for WebDataset shards"""
    
    def __init__(self, output_dir: Path, prefix: str, 
                max_shard_size_mb: int = MAX_SHARD_SIZE_MB):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.max_shard_size_bytes = max_shard_size_mb * 1024 * 1024
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track current shard info
        self.current_shard_index = self._find_next_shard_index()
        self.current_shard_path = self._get_shard_path(self.current_shard_index)
        self.current_shard_size = 0
        self.current_tar = None
        self.sample_count = 0
        self.video_count = 0
        self.image_count = 0
        self.captioned_count = 0
        
        # Set of sample keys in current shard
        self.current_sample_keys = set()
        
        # Metadata for current shard
        self.metadata = {
            "prefix": prefix,
            "index": self.current_shard_index,
            "sample_count": 0,
            "video_count": 0,
            "image_count": 0,
            "captioned_count": 0,
            "creation_time": None,
            "samples": {}
        }
        
        # Open first shard
        self._open_shard()
    
    def _find_next_shard_index(self) -> int:
        """Find the next available shard index"""
        existing_shards = list(self.output_dir.glob(f"{self.prefix}-*.tar"))
        if not existing_shards:
            return 0
        
        # Extract shard indices from filenames
        indices = []
        for shard in existing_shards:
            try:
                # Extract the index part from the filename
                index_part = shard.stem.split('-')[-1]
                if index_part.isdigit():
                    indices.append(int(index_part))
            except (ValueError, IndexError):
                continue
        
        return max(indices) + 1 if indices else 0
    
    def _get_shard_path(self, index: int) -> Path:
        """Generate path for a shard with the given index"""
        return self.output_dir / SHARD_PATTERN.format(prefix=self.prefix, number=index)
    
    def _open_shard(self):
        """Open a new shard for writing"""
        if self.current_tar:
            self._close_shard()
        
        logger.info(f"Opening new shard: {self.current_shard_path}")
        self.current_tar = tarfile.open(self.current_shard_path, 'w')
        self.current_shard_size = 0
        self.current_sample_keys = set()
        
        # Initialize metadata for the new shard
        self.metadata = {
            "prefix": self.prefix,
            "index": self.current_shard_index,
            "sample_count": 0,
            "video_count": 0,
            "image_count": 0,
            "captioned_count": 0,
            "creation_time": None,
            "samples": {}
        }
    
    def _add_file_to_tar(self, tar_path: str, content: bytes) -> int:
        """Add a file to the tar archive and return its size"""
        info = tarfile.TarInfo(name=tar_path)
        info.size = len(content)
        info.mtime = int(datetime.now().timestamp())
        
        # Use a BytesIO object as a file-like object
        file_obj = io.BytesIO(content)
        self.current_tar.addfile(info, file_obj)
        
        return len(content)
    
    def _close_shard(self):
        """Close the current shard and write metadata"""
        if not self.current_tar:
            return
        
        # Update metadata timestamp
        self.metadata["creation_time"] = datetime.now().isoformat()
        
        # Add metadata to shard
        metadata_content = json.dumps(self.metadata, indent=2).encode('utf-8')
        self._add_file_to_tar(METADATA_FILENAME, metadata_content)
        
        # Close the tar file
        self.current_tar.close()
        self.current_tar = None
        
        # Log shard info
        logger.info(f"Closed shard {self.current_shard_path}: {self.metadata['sample_count']} samples, "
                  f"{self.metadata['video_count']} videos, {self.metadata['image_count']} images")
    
    def add_sample(self, key: str, type_name: str, extension: str, content: bytes) -> bool:
        """Add a file to the current sample in the shard
        
        Args:
            key: Sample key
            type_name: Type of file ("video", "image", "caption", "data")
            extension: File extension without leading dot
            content: File content as bytes
            
        Returns:
            True if file was added successfully
        """
        # Check if we need to start a new shard
        estimated_size = len(content) + 1024  # Add some overhead for tar headers
        if (self.current_shard_size + estimated_size > self.max_shard_size_bytes or
            len(self.current_sample_keys) >= TARGET_SAMPLES_PER_SHARD or
            (key in self.current_sample_keys and type_name in ["video", "image"])):
            # Close current shard and open a new one
            self._close_shard()
            self.current_shard_index += 1
            self.current_shard_path = self._get_shard_path(self.current_shard_index)
            self._open_shard()
        
        # Add file to shard
        tar_path = f"{key}.{extension}"
        size = self._add_file_to_tar(tar_path, content)
        self.current_shard_size += size
        
        # Update metadata
        if key not in self.metadata["samples"]:
            self.metadata["samples"][key] = {
                "files": [],
                "has_caption": False,
                "type": type_name
            }
            self.metadata["sample_count"] += 1
            
            # Update counters
            if type_name == "video":
                self.metadata["video_count"] += 1
                self.video_count += 1
            elif type_name == "image":
                self.metadata["image_count"] += 1
                self.image_count += 1
        
        self.metadata["samples"][key]["files"].append({
            "path": tar_path,
            "ext": extension,
            "size": len(content)
        })
        
        # Track if this sample has a caption
        if type_name == "caption":
            self.metadata["samples"][key]["has_caption"] = True
            if not self.metadata["samples"][key].get("has_caption_before", False):
                self.metadata["captioned_count"] += 1
                self.metadata["samples"][key]["has_caption_before"] = True
                self.captioned_count += 1
        
        # Add to current sample keys set
        self.current_sample_keys.add(key)
        
        return True
    
    def add_file(self, file_path: Path, sample_key: Optional[str] = None) -> bool:
        """Add a file from disk to the shard
        
        Args:
            file_path: Path to the file
            sample_key: Optional sample key (defaults to file stem)
            
        Returns:
            True if file was added successfully
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            # Determine sample key
            key = sample_key or file_path.stem
            
            # Read file content
            content = file_path.read_bytes()
            
            # Determine file type from extension
            ext = file_path.suffix.lower().lstrip('.')
            
            # Determine type name
            type_name = "data"  # Default
            for name, exts in EXTENSIONS.items():
                if f".{ext}" in exts:
                    type_name = name
                    break
            
            # Add to shard
            return self.add_sample(key, type_name, ext, content)
            
        except Exception as e:
            logger.error(f"Error adding file {file_path} to shard: {e}")
            return False
    
    def close(self):
        """Close the writer and finish the current shard"""
        self._close_shard()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the written shards"""
        return {
            "shard_count": self.current_shard_index + 1,
            "sample_count": self.sample_count,
            "video_count": self.video_count,
            "image_count": self.image_count,
            "captioned_count": self.captioned_count,
            "current_shard_path": str(self.current_shard_path),
            "current_shard_size": self.current_shard_size
        }
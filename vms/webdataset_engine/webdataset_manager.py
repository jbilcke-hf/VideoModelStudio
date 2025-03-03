"""
Core WebDataset functionality for Video Model Studio
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
import webdataset as wds
import torch
from torch.utils.data import DataLoader
import torch.utils.data
import numpy as np
import cv2
from PIL import Image
import io

from .config_webdataset import (
    SHARD_PATTERN, MAX_SHARD_SIZE_MB, TARGET_SAMPLES_PER_SHARD,
    METADATA_FILENAME, SHARD_DIRS, EXTENSIONS, BUFFER_SIZE, PREFETCH_FACTOR,
    CACHE_ENABLED, MAX_CACHE_SIZE_GB, CACHE_EXPIRATION_DAYS, get_shard_path, get_all_shards
)

from ..config import STORAGE_PATH, TRAINING_PATH, NORMALIZE_IMAGES_TO, JPEG_QUALITY

logger = logging.getLogger(__name__)

class WebDatasetManager:
    """Manager for WebDataset operations"""
    
    def __init__(self, storage_path: Path = STORAGE_PATH):
        self.storage_path = storage_path
        self.current_shard_info = {}
        self._setup_directories()
        
    def _setup_directories(self):
        """Ensure all required directories exist"""
        # Create storage path if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create shard directories
        for dirname in SHARD_DIRS.values():
            (self.storage_path / dirname).mkdir(parents=True, exist_ok=True)
    
    def get_shard_directory(self, shard_type: str) -> Path:
        """Get path to a specific shard directory"""
        if shard_type not in SHARD_DIRS:
            raise ValueError(f"Unknown shard type: {shard_type}")
        return self.storage_path / SHARD_DIRS[shard_type]
    
    def list_shards(self, shard_type: str, prefix: str = "") -> List[Path]:
        """List all shards of a specific type"""
        shard_dir = self.get_shard_directory(shard_type)
        if prefix:
            pattern = f"{prefix}-*.tar"
        else:
            pattern = "*.tar"
        return sorted(shard_dir.glob(pattern))
    
    def get_shard_metadata(self, shard_path: Path) -> Dict:
        """Extract metadata from a shard"""
        try:
            with tarfile.open(shard_path, 'r') as tar:
                # Look for metadata file
                metadata_member = None
                for member in tar.getmembers():
                    if member.name == METADATA_FILENAME:
                        metadata_member = member
                        break
                
                if metadata_member:
                    f = tar.extractfile(metadata_member)
                    if f:
                        metadata_content = f.read().decode('utf-8')
                        return json.loads(metadata_content)
            
            # If no metadata found, create basic metadata with sample count
            with tarfile.open(shard_path, 'r') as tar:
                members = tar.getmembers()
                
                # Group members by prefix to count samples
                samples = {}
                video_count = 0
                image_count = 0
                captioned_count = 0
                
                for member in members:
                    if member.isfile():
                        # Extract the key (everything up to the first dot after the last slash)
                        path = Path(member.name)
                        key = path.stem
                        ext = path.suffix.lower()
                        
                        if key not in samples:
                            samples[key] = {
                                "files": [],
                                "has_caption": False,
                                "type": "unknown"
                            }
                        
                        # Determine file type
                        if ext in ['.mp4', '.webm']:
                            samples[key]["type"] = "video"
                        elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.avif']:
                            if samples[key]["type"] != "video":  # Don't overwrite video type
                                samples[key]["type"] = "image"
                        elif ext in ['.txt', '.caption', '.json', '.cls']:
                            samples[key]["has_caption"] = True
                        
                        # Add file to sample
                        samples[key]["files"].append({
                            "path": member.name,
                            "ext": ext,
                            "size": member.size
                        })
                
                # Count sample types
                for key, info in samples.items():
                    if info["type"] == "video":
                        video_count += 1
                    elif info["type"] == "image":
                        image_count += 1
                    if info["has_caption"]:
                        captioned_count += 1
                
                # Create basic metadata
                return {
                    "sample_count": len(samples),
                    "video_count": video_count,
                    "image_count": image_count,
                    "captioned_count": captioned_count,
                    "samples": samples
                }
                
        except Exception as e:
            logger.error(f"Error reading metadata from {shard_path}: {e}")
            return {}
    
    def get_dataset_stats(self, shard_type: str, prefix: str = "") -> Dict:
        """Get statistics about a dataset (shards, samples, etc.)"""
        shards = self.list_shards(shard_type, prefix)
        
        stats = {
            "shard_count": len(shards),
            "total_size_bytes": 0,
            "sample_count": 0,
            "video_count": 0,
            "image_count": 0,
            "has_captions": 0,
            "largest_shard_bytes": 0,
            "smallest_shard_bytes": float('inf') if shards else 0,
        }
        
        for shard in shards:
            size = shard.stat().st_size
            stats["total_size_bytes"] += size
            stats["largest_shard_bytes"] = max(stats["largest_shard_bytes"], size)
            if size > 0:
                stats["smallest_shard_bytes"] = min(stats["smallest_shard_bytes"], size)
            
            # Extract metadata for more detailed stats
            metadata = self.get_shard_metadata(shard)
            if metadata:
                stats["sample_count"] += metadata.get("sample_count", 0)
                stats["video_count"] += metadata.get("video_count", 0)
                stats["image_count"] += metadata.get("image_count", 0)
                stats["has_captions"] += metadata.get("captioned_count", 0)
        
        return stats
    
    def extract_sample(self, shard_path: Path, sample_key: str, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Extract a specific sample from a shard
        
        Args:
            shard_path: Path to the shard file
            sample_key: Key of the sample to extract
            output_dir: Directory to extract files to, uses temp dir if None
            
        Returns:
            Dictionary mapping extensions to file paths
        """
        result = {}
        temp_dir = None
        
        try:
            # Create temp dir if no output dir specified
            if output_dir is None:
                temp_dir = tempfile.TemporaryDirectory()
                output_dir = Path(temp_dir.name)
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Open the shard
            with tarfile.open(shard_path, 'r') as tar:
                # Find all files with the matching sample key
                for member in tar.getmembers():
                    path = Path(member.name)
                    key = path.stem
                    
                    if key == sample_key:
                        # Extract the file
                        tar.extract(member, output_dir)
                        extracted_path = output_dir / member.name
                        
                        # Add to result dictionary
                        result[path.suffix] = extracted_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting sample {sample_key} from {shard_path}: {e}")
            return {}
        finally:
            # Clean up temp dir if we created one and there was an error
            if temp_dir is not None and not result:
                temp_dir.cleanup()
    
    def create_shard_writer(self, shard_type: str, prefix: str) -> 'ShardWriter':
        """Create a new shard writer for the specified type
        
        Args:
            shard_type: Type of shard to create ("raw", "processed", "training")
            prefix: Prefix for shard filenames
            
        Returns:
            ShardWriter instance
        """
        from .shard_writer import ShardWriter
        shard_dir = self.get_shard_directory(shard_type)
        return ShardWriter(shard_dir, prefix)
    
    def create_dataset(self, shard_type: str, prefix: str = "", 
                       shuffle: bool = True, pipeline=None) -> wds.WebDataset:
        """Create a WebDataset from shards
        
        Args:
            shard_type: Type of shards to use ("raw", "processed", "training")
            prefix: Optional prefix to filter shards
            shuffle: Whether to shuffle shards
            pipeline: Custom processing pipeline
            
        Returns:
            webdataset.WebDataset instance
        """
        shards = self.list_shards(shard_type, prefix)
        if not shards:
            raise ValueError(f"No shards found for type '{shard_type}' with prefix '{prefix}'")
        
        # Convert to URLs compatible with WebDataset
        urls = [f"file:{shard}" for shard in shards]
        
        # Create base dataset
        dataset = wds.WebDataset(urls, shardshuffle=shuffle)
        
        # Apply default preprocessing pipeline or custom one
        if pipeline:
            dataset = pipeline(dataset)
        else:
            # Apply default pipeline
            dataset = (
                dataset
                .decode("pil")
                .rename(image="jpg;png;jpeg;webp", video="mp4;webm", caption="txt;caption;json;cls")
                .to_tuple("image;video", "caption")
            )
        
        return dataset
    
    def create_dataloader(self, dataset: wds.WebDataset, batch_size: int = 1, 
                         num_workers: int = 4) -> DataLoader:
        """Create a DataLoader from a WebDataset
        
        Args:
            dataset: WebDataset instance
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            torch.utils.data.DataLoader instance
        """
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=PREFETCH_FACTOR,
        )
    
    def get_sample(self, shard_path: Path, sample_key: str) -> Dict[str, Any]:
        """Get a specific sample from a shard without extracting to disk
        
        Args:
            shard_path: Path to the shard
            sample_key: Key of the sample to get
            
        Returns:
            Sample dictionary with file contents
        """
        try:
            # Create dataset for this shard
            url = f"file:{shard_path}"
            dataset = wds.WebDataset(url)
            
            # Find the sample with matching key
            for sample in dataset:
                if sample.get("__key__") == sample_key:
                    return sample
            
            # Sample not found
            return {}
            
        except Exception as e:
            logger.error(f"Error getting sample {sample_key} from {shard_path}: {e}")
            return {}
    
    def export_to_training_dir(self, output_dir: Path, shard_type: str = "training", 
                             prefix: str = "") -> Tuple[int, int]:
        """Export samples from WebDataset shards to a flat directory structure
        for training with frameworks that don't support WebDataset
        
        Args:
            output_dir: Directory to export files to
            shard_type: Type of shards to export from
            prefix: Optional prefix to filter shards
            
        Returns:
            Tuple of (exported_samples, total_bytes)
        """
        exported_samples = 0
        total_bytes = 0
        
        # Make sure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all shards of the specified type
        shards = self.list_shards(shard_type, prefix)
        
        try:
            for shard_path in shards:
                # Create dataset for this shard
                url = f"file:{shard_path}"
                dataset = wds.WebDataset(url)
                
                # Process each sample
                for sample in dataset:
                    sample_key = sample.get("__key__")
                    if not sample_key:
                        continue
                    
                    # Create safe filename from key
                    safe_key = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in sample_key)
                    
                    # Export each file in the sample
                    has_media = False
                    caption_content = None
                    
                    for key, value in sample.items():
                        if key == "__key__":
                            continue
                        
                        # Handle caption files
                        if key in ["txt", "caption", "json", "cls"]:
                            if isinstance(value, bytes):
                                caption_content = value.decode('utf-8')
                            else:
                                caption_content = str(value)
                        
                        # Handle media files
                        elif key in ["jpg", "jpeg", "png", "webp", "mp4", "webm"]:
                            file_path = output_dir / f"{safe_key}.{key}"
                            
                            # Write file
                            if isinstance(value, bytes):
                                with open(file_path, 'wb') as f:
                                    f.write(value)
                                total_bytes += len(value)
                                has_media = True
                            elif isinstance(value, (PIL.Image.Image, np.ndarray)):
                                # Convert PIL image to bytes
                                if isinstance(value, PIL.Image.Image):
                                    img_byte_array = io.BytesIO()
                                    value.save(img_byte_array, format='PNG')
                                    img_bytes = img_byte_array.getvalue()
                                    with open(file_path, 'wb') as f:
                                        f.write(img_bytes)
                                    total_bytes += len(img_bytes)
                                # Convert numpy array to bytes
                                else:
                                    cv2.imwrite(str(file_path), value)
                                    total_bytes += file_path.stat().st_size
                                has_media = True
                    
                    # Write caption file if we have both media and caption
                    if has_media and caption_content:
                        caption_path = output_dir / f"{safe_key}.txt"
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(caption_content)
                        exported_samples += 1
                    elif has_media:
                        exported_samples += 1
        
        except Exception as e:
            logger.error(f"Error exporting samples to {output_dir}: {e}")
        
        return exported_samples, total_bytes
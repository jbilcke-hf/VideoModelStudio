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

class WebDatasetProcessor:
    """Processor for streaming transformations on WebDatasets"""
    
    @staticmethod
    def remove_black_bars(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Remove black bars from videos or images in a sample"""
        if "__key__" not in sample:
            return sample
        
        # Process images
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            key = ext[1:]  # Remove the dot
            if key in sample:
                try:
                    # Convert to numpy array for processing
                    img_data = sample[key]
                    if isinstance(img_data, bytes):
                        img = Image.open(io.BytesIO(img_data))
                        img_np = np.array(img)
                    else:
                        img_np = np.array(img_data)
                    
                    # Detect and remove black bars
                    top, bottom, left, right = WebDatasetProcessor._detect_black_bars(img_np)
                    
                    # Crop if black bars detected
                    if any([top > 0, bottom < img_np.shape[0] - 1, 
                            left > 0, right < img_np.shape[1] - 1]):
                        cropped = img_np[top:bottom, left:right]
                        
                        # Convert back to bytes
                        img = Image.fromarray(cropped)
                        buffer = io.BytesIO()
                        if NORMALIZE_IMAGES_TO == 'png':
                            img.save(buffer, 'PNG', optimize=True)
                        else:  # jpg
                            img.save(buffer, 'JPEG', quality=JPEG_QUALITY, optimize=True)
                        sample[key] = buffer.getvalue()
                except Exception as e:
                    logger.error(f"Error processing image in sample {sample['__key__']}: {e}")
        
        # TODO: Add video processing with ffmpeg when needed
        
        return sample
    
    @staticmethod
    def _detect_black_bars(img: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect black bars in image"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Threshold to detect black regions
        threshold = 20
        black_mask = gray < threshold
        
        # Find black bars by analyzing row/column means
        row_means = np.mean(black_mask, axis=1)
        col_means = np.mean(black_mask, axis=0)
        
        # Detect edges where black bars end (95% threshold)
        black_threshold = 0.95
        
        # Find top and bottom crops
        top = 0
        bottom = img.shape[0]
        
        for i, mean in enumerate(row_means):
            if mean > black_threshold:
                top = i + 1
            else:
                break
                
        for i, mean in enumerate(reversed(row_means)):
            if mean > black_threshold:
                bottom = img.shape[0] - i - 1
            else:
                break
        
        # Find left and right crops
        left = 0
        right = img.shape[1]
        
        for i, mean in enumerate(col_means):
            if mean > black_threshold:
                left = i + 1
            else:
                break
                
        for i, mean in enumerate(reversed(col_means)):
            if mean > black_threshold:
                right = img.shape[1] - i - 1
            else:
                break
                
        return top, bottom, left, right
    
    @staticmethod
    def create_pipeline(*transforms):
        """Create a pipeline of transformations for WebDataset"""
        def apply_pipeline(dataset):
            for transform in transforms:
                dataset = dataset.map(transform)
            return dataset
        return apply_pipeline
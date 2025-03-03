"""
WebDataset format handling for Video Model Studio
"""

import os
import tarfile
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from ..utils import is_image_file, is_video_file, extract_scene_info

logger = logging.getLogger(__name__)

def is_webdataset_file(file_path: Path) -> bool:
    """Check if file is a WebDataset tar file
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if file has .tar extension
    """
    return file_path.suffix.lower() == '.tar'

def process_webdataset_shard(
    tar_path: Path, 
    videos_output_dir: Path, 
    staging_output_dir: Path
) -> Tuple[int, int]:
    """Process a WebDataset shard (tar file) extracting video/image and caption pairs
    
    Args:
        tar_path: Path to the WebDataset tar file
        videos_output_dir: Directory to store videos for splitting
        staging_output_dir: Directory to store images and captions
        
    Returns:
        Tuple of (video_count, image_count)
    """
    video_count = 0
    image_count = 0
    
    try:
        # Dictionary to store grouped files by prefix
        grouped_files = {}
        
        # First pass: collect and group files by prefix
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isdir():
                    continue
                    
                # Skip hidden files
                if os.path.basename(member.name).startswith('.'):
                    continue
                
                # Extract file prefix (everything up to the first dot after the last slash)
                file_path = Path(member.name)
                file_name = file_path.name
                
                # Get prefix (filename without extensions)
                # For WebDataset, the prefix is everything up to the first dot
                prefix_parts = file_name.split('.', 1)
                if len(prefix_parts) < 2:
                    # No extension, skip
                    continue
                
                prefix = prefix_parts[0]
                extension = '.' + prefix_parts[1]
                
                # Include directory in the prefix to keep samples grouped correctly
                full_prefix = str(file_path.parent / prefix) if file_path.parent != Path('.') else prefix
                
                if full_prefix not in grouped_files:
                    grouped_files[full_prefix] = []
                
                grouped_files[full_prefix].append((member, extension))
        
        # Second pass: extract and process grouped files
        with tarfile.open(tar_path, 'r') as tar:
            for prefix, members in grouped_files.items():
                # Create safe filename from prefix
                safe_prefix = Path(prefix).name
                
                # Find media and caption files
                media_file = None
                caption_file = None
                media_ext = None
                
                for member, ext in members:
                    if ext.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.avif', '.heic']:
                        media_file = member
                        media_ext = ext
                    elif ext.lower() in ['.mp4', '.webm']:
                        media_file = member
                        media_ext = ext
                    elif ext.lower() in ['.txt', '.caption', '.json', '.cls']:
                        caption_file = member
                
                # If we have a media file, process it
                if media_file:
                    # Determine if it's video or image
                    is_video = media_ext.lower() in ['.mp4', '.webm']
                    
                    # Choose target directory based on media type
                    target_dir = videos_output_dir if is_video else staging_output_dir
                    
                    # Create target filename
                    target_filename = f"{safe_prefix}{media_ext}"
                    target_path = target_dir / target_filename
                    
                    # If file already exists, add number suffix
                    counter = 1
                    while target_path.exists():
                        target_path = target_dir / f"{safe_prefix}___{counter}{media_ext}"
                        counter += 1
                    
                    # Extract media file
                    with open(target_path, 'wb') as f:
                        f.write(tar.extractfile(media_file).read())
                    
                    # If we have a caption file, extract it too
                    if caption_file:
                        caption_text = tar.extractfile(caption_file).read().decode('utf-8', errors='ignore')
                        
                        # Save caption with media file extension
                        caption_path = target_path.with_suffix('.txt')
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(caption_text)
                    
                    # Update counters
                    if is_video:
                        video_count += 1
                    else:
                        image_count += 1
    
    except Exception as e:
        logger.error(f"Error processing WebDataset file {tar_path}: {e}")
        raise
    
    return video_count, image_count
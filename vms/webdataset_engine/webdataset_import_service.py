"""
Import service for Video Model Studio using WebDataset
"""

import os
import tarfile
import tempfile
import zipfile
import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import gradio as gr
from pytubefix import YouTube

from .webdataset_manager import WebDatasetManager
from .shard_writer import ShardWriter
from .config_webdataset import EXTENSIONS

from ..config import (
    STORAGE_PATH, DEFAULT_PROMPT_PREFIX, NORMALIZE_IMAGES_TO, VIDEOS_TO_SPLIT_PATH, STAGING_PATH
)
from ..utils import (
    is_image_file, is_video_file, 
    normalize_image, get_video_fps,
    add_prefix_to_caption
)

logger = logging.getLogger(__name__)

class WebDatasetImportService:
    """Service for importing various file formats into WebDataset shards"""
    
    def __init__(self):
        self.wds_manager = WebDatasetManager(STORAGE_PATH)
    
    def process_uploaded_files(self, file_paths: List[str]) -> str:
        """Process uploaded files (ZIP, TAR, video, image) converting to WebDataset
        
        Args:
            file_paths: File paths to the uploaded files from Gradio
                
        Returns:
            Status message string
        """
        if not file_paths:
            return "No files uploaded"
        
        video_count = 0
        image_count = 0
        tar_count = 0
        
        for file_path in file_paths:
            try:
                file_path = Path(file_path)
                file_ext = file_path.suffix.lower()
                
                if file_ext == '.zip':
                    # Process ZIP containing multiple files
                    vc, ic, tc = self._process_zip_file(file_path)
                    video_count += vc
                    image_count += ic
                    tar_count += tc
                
                elif file_ext == '.tar':
                    # Direct import of WebDataset shard
                    result = self._import_webdataset_shard(file_path)
                    if result:
                        tar_count += 1
                
                elif file_ext in ['.mp4', '.webm']:
                    # Process single video file
                    result = self._process_video_file(file_path)
                    if result:
                        video_count += 1
                
                elif is_image_file(file_path):
                    # Process single image file
                    result = self._process_image_file(file_path)
                    if result:
                        image_count += 1
                
                else:
                    logger.warning(f"Unsupported file type: {file_ext}")
                    raise gr.Error(f"Unsupported file type: {file_ext}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                raise gr.Error(f"Error processing file: {str(e)}")
        
        # Generate status message
        parts = []
        if tar_count > 0:
            parts.append(f"{tar_count} WebDataset shard{'s' if tar_count > 1 else ''}")
        if video_count > 0:
            parts.append(f"{video_count} video{'s' if video_count > 1 else ''}")
        if image_count > 0:
            parts.append(f"{image_count} image{'s' if image_count > 1 else ''}")
        
        if not parts:
            return "No supported media files found"
        
        status = f"Successfully imported {', '.join(parts)}"
        gr.Info(status)
        return status
    
    def _process_zip_file(self, file_path: Path) -> Tuple[int, int, int]:
        """Process uploaded ZIP file containing media or WebDataset files
        
        Args:
            file_path: Path to the uploaded ZIP file
                
        Returns:
            Tuple of (video_count, image_count, tar_count)
        """
        video_count = 0
        image_count = 0
        tar_count = 0
        
        # Create unique prefix for this import
        import_prefix = f"import_{uuid.uuid4().hex[:8]}"
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            extract_dir = Path(temp_dir) / "extracted"
            extract_dir.mkdir()
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Process WebDataset tar files first
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.startswith('._'):  # Skip Mac metadata
                        continue
                    
                    file_path = Path(root) / file
                    
                    if file.lower().endswith('.tar'):
                        # Import WebDataset shard
                        result = self._import_webdataset_shard(file_path)
                        if result:
                            tar_count += 1
            
            # Then process regular media files
            media_files = []
            
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.startswith('._'):  # Skip Mac metadata
                        continue
                    
                    file_path = Path(root) / file
                    
                    if file.lower().endswith('.tar'):
                        continue  # Already processed
                    
                    if is_video_file(file_path) or is_image_file(file_path):
                        media_files.append(file_path)
            
            # Process media files
            if media_files:
                # Create a shard writer for batch import
                raw_dir = self.wds_manager.get_shard_directory("raw")
                writer = ShardWriter(raw_dir, import_prefix)
                
                try:
                    for file_path in media_files:
                        # Generate a unique key for this sample
                        sample_key = f"{file_path.stem}_{uuid.uuid4().hex[:6]}"
                        
                        # Determine file type and read content
                        if is_video_file(file_path):
                            content = file_path.read_bytes()
                            ext = file_path.suffix.lstrip('.')
                            writer.add_sample(sample_key, "video", ext, content)
                            video_count += 1
                            
                            # Check for caption file
                            caption_path = file_path.with_suffix('.txt')
                            if caption_path.exists():
                                caption = caption_path.read_text()
                                caption = add_prefix_to_caption(caption, DEFAULT_PROMPT_PREFIX)
                                writer.add_sample(sample_key, "caption", "txt", caption.encode('utf-8'))
                            
                        elif is_image_file(file_path):
                            # Process image through normalization
                            with tempfile.NamedTemporaryFile(suffix=f".{NORMALIZE_IMAGES_TO}") as tmp:
                                temp_path = Path(tmp.name)
                                if normalize_image(file_path, temp_path):
                                    content = temp_path.read_bytes()
                                    writer.add_sample(sample_key, "image", NORMALIZE_IMAGES_TO, content)
                                    image_count += 1
                                    
                                    # Check for caption file
                                    caption_path = file_path.with_suffix('.txt')
                                    if caption_path.exists():
                                        caption = caption_path.read_text()
                                        caption = add_prefix_to_caption(caption, DEFAULT_PROMPT_PREFIX)
                                        writer.add_sample(sample_key, "caption", "txt", caption.encode('utf-8'))
                
                finally:
                    writer.close()
        
        return video_count, image_count, tar_count
    
    def _import_webdataset_shard(self, file_path: Path) -> bool:
        """Import a WebDataset shard directly
        
        Args:
            file_path: Path to the WebDataset tar file
                
        Returns:
            bool: Success status
        """
        try:
            # Verify it's a valid tar file
            with tarfile.open(file_path, 'r') as _:
                pass  # Just check it opens
            
            # Copy to raw shards directory
            dest_dir = self.wds_manager.get_shard_directory("raw")
            dest_path = dest_dir / f"import_{uuid.uuid4().hex[:8]}_{file_path.name}"
            shutil.copy2(file_path, dest_path)
            
            logger.info(f"Imported WebDataset shard: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing WebDataset shard {file_path}: {e}")
            return False
    
    def _process_video_file(self, file_path: Path) -> bool:
        """Process a single video file and add to WebDataset
        
        Args:
            file_path: Path to the video file
                
        Returns:
            bool: Success status
        """
        try:
            # Generate a unique sample key
            sample_key = f"{file_path.stem}_{uuid.uuid4().hex[:6]}"
            
            # Also copy the original file to VIDEOS_TO_SPLIT_PATH for compatibility with the existing system
            target_path = VIDEOS_TO_SPLIT_PATH / file_path.name
            
            # If file already exists, add number suffix
            counter = 1
            while target_path.exists():
                target_path = VIDEOS_TO_SPLIT_PATH / f"{file_path.stem}_{counter}{file_path.suffix}"
                counter += 1
                
            shutil.copy2(file_path, target_path)
            
            # Create a shard writer
            raw_dir = self.wds_manager.get_shard_directory("raw")
            writer = ShardWriter(raw_dir, f"video_{uuid.uuid4().hex[:8]}")
            
            try:
                # Read video content
                content = file_path.read_bytes()
                ext = file_path.suffix.lstrip('.')
                
                # Add to shard
                writer.add_sample(sample_key, "video", ext, content)
                
                # Check for caption file
                caption_path = file_path.with_suffix('.txt')
                if caption_path.exists():
                    caption = caption_path.read_text()
                    caption = add_prefix_to_caption(caption, DEFAULT_PROMPT_PREFIX)
                    writer.add_sample(sample_key, "caption", "txt", caption.encode('utf-8'))
                
                logger.info(f"Added video {file_path.name} to WebDataset")
                gr.Info(f"Successfully imported video: {file_path.name}")
                return True
                
            finally:
                writer.close()
                
        except Exception as e:
            logger.error(f"Error processing video file {file_path}: {e}")
            return False
    
    def _process_image_file(self, file_path: Path) -> bool:
        """Process a single image file and add to WebDataset
        
        Args:
            file_path: Path to the image file
                
        Returns:
            bool: Success status
        """
        try:
            # Generate a unique sample key
            sample_key = f"{file_path.stem}_{uuid.uuid4().hex[:6]}"
            
            # Also copy to STAGING_PATH for compatibility with existing system
            target_path = STAGING_PATH / f"{file_path.stem}.{NORMALIZE_IMAGES_TO}"
            
            # Create a shard writer
            raw_dir = self.wds_manager.get_shard_directory("raw")
            writer = ShardWriter(raw_dir, f"image_{uuid.uuid4().hex[:8]}")
            
            try:
                # Process image through normalization
                with tempfile.NamedTemporaryFile(suffix=f".{NORMALIZE_IMAGES_TO}") as tmp:
                    temp_path = Path(tmp.name)
                    if normalize_image(file_path, temp_path):
                        # Copy to staging path
                        shutil.copy2(temp_path, target_path)
                        
                        # Add to WebDataset
                        content = temp_path.read_bytes()
                        writer.add_sample(sample_key, "image", NORMALIZE_IMAGES_TO, content)
                        
                        # Check for caption file
                        caption_path = file_path.with_suffix('.txt')
                        if caption_path.exists():
                            caption = caption_path.read_text()
                            caption = add_prefix_to_caption(caption, DEFAULT_PROMPT_PREFIX)
                            writer.add_sample(sample_key, "caption", "txt", caption.encode('utf-8'))
                            
                            # Also copy caption to staging
                            target_caption_path = target_path.with_suffix('.txt')
                            target_caption_path.write_text(caption)
                        
                        logger.info(f"Added image {file_path.name} to WebDataset")
                        gr.Info(f"Successfully imported image: {file_path.name}")
                        return True
                    
                    return False
                    
            finally:
                writer.close()
                
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
            return False
    
    def download_youtube_video(self, url: str, progress=None) -> str:
        """Download a video from YouTube directly to WebDataset
        
        Args:
            url: YouTube video URL
            progress: Optional Gradio progress indicator
            
        Returns:
            Status message
        """
        try:
            # Extract video ID and create YouTube object
            yt = YouTube(url, on_progress_callback=lambda stream, chunk, bytes_remaining: 
                progress((1 - bytes_remaining / stream.filesize), desc="Downloading...")
                if progress else None)
            
            video_id = yt.video_id
            
            # Download highest quality progressive MP4
            if progress:
                progress(0, desc="Getting video streams...")
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not video:
                gr.Error("Could not find a compatible video format")
                return "Could not find a compatible video format"
            
            # Create temporary file for downloading
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Download the video
                if progress:
                    progress(0, desc="Starting download...")
                
                video.download(output_path=os.path.dirname(temp_path), filename=os.path.basename(temp_path))
                
                if progress:
                    progress(1, desc="Download complete, adding to WebDataset...")
                
                # Process downloaded video (add to WebDataset)
                result = self._process_video_file(temp_path)
                
                # Clean up temp file
                temp_path.unlink()
                
                if not result:
                    return f"Error processing downloaded video: {yt.title}"
                
                # Update UI
                if progress:
                    progress(1, desc="Video added to dataset!")
                gr.Info("YouTube video successfully added to dataset")
                return f"Successfully downloaded and imported video: {yt.title}"
                
        except Exception as e:
            gr.Error(f"Error downloading video: {str(e)}")
            return f"Error downloading video: {str(e)}"
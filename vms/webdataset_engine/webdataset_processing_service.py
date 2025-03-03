"""
Processing service for scene detection and video preprocessing with WebDataset
"""

import os
import io
import tempfile
import asyncio
import logging
import uuid
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any, AsyncGenerator
import cv2
import numpy as np
import gradio as gr

from scenedetect import detect, ContentDetector, SceneManager, open_video
from scenedetect.video_splitter import split_video_ffmpeg
import webdataset as wds

from .webdataset_manager import WebDatasetManager
from .shard_writer import ShardWriter
from .config_webdataset import get_shard_path

from ..config import (
    STORAGE_PATH, NORMALIZE_IMAGES_TO, JPEG_QUALITY, 
    DEFAULT_PROMPT_PREFIX, VIDEOS_TO_SPLIT_PATH, STAGING_PATH
)
from ..utils import get_video_fps, add_prefix_to_caption

logger = logging.getLogger(__name__)

class WebDatasetProcessingService:
    """Service for scene detection and video preprocessing within WebDataset"""
    
    def __init__(self):
        self.wds_manager = WebDatasetManager(STORAGE_PATH)
        self.processing = False
        self._current_file = None
        self._scene_counts = {}
        self._processing_status = {}
    
    def is_processing(self) -> bool:
        """Check if processing is currently active"""
        return self.processing
    
    def get_current_file(self) -> Optional[str]:
        """Get the currently processing file name"""
        return self._current_file
    
    def get_processing_status(self, key: str) -> str:
        """Get processing status for a specific sample"""
        return self._processing_status.get(key, "not processed")
    
    def get_scene_count(self, key: str) -> Optional[int]:
        """Get number of scenes detected for a video"""
        return self._scene_counts.get(key)
    
    def list_unprocessed_videos(self) -> List[List[str]]:
        """List videos in raw shards that need scene detection
        
        Returns:
            List of [sample_key, status] pairs
        """
        results = []
        
        # List all raw shards
        raw_shards = self.wds_manager.list_shards("raw")
        
        for shard_path in raw_shards:
            # Extract metadata to find videos
            metadata = self.wds_manager.get_shard_metadata(shard_path)
            
            if not metadata or "samples" not in metadata:
                continue
            
            # Find video samples
            for key, info in metadata.get("samples", {}).items():
                if info.get("type") == "video":
                    status = self.get_processing_status(key)
                    results.append([key, status])
        
        # Also list videos from VIDEOS_TO_SPLIT_PATH for compatibility with existing system
        for video_file in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
            status = self.get_processing_status(video_file.name)
            results.append([video_file.name, status])
            
        # Sort and remove duplicates
        unique_results = {}
        for key, status in results:
            unique_results[key] = status
        
        return sorted([[key, status] for key, status in unique_results.items()], key=lambda x: x[0])
    
    async def process_video_sample(self, sample: Dict[str, Any], enable_splitting: bool) -> int:
        """Process a video sample from a WebDataset to detect and split scenes
        
        Args:
            sample: WebDataset sample containing video
            enable_splitting: Whether to split videos into scenes
            
        Returns:
            Number of scenes detected (0 means no splitting needed)
        """
        if "__key__" not in sample:
            return 0
        
        key = sample["__key__"]
        self._current_file = key
        self._processing_status[key] = f'Processing video "{key}"...'
        
        # Find video content
        video_data = None
        video_ext = None
        
        for ext in ['.mp4', '.webm']:
            ext_key = ext[1:]  # Remove the dot
            if ext_key in sample:
                video_data = sample[ext_key]
                video_ext = ext_key
                break
        
        if not video_data:
            self._processing_status[key] = "No video content found"
            return 0
        
        # Create temporary directory and files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Write video to temp file
            video_path = temp_dir / f"video.{video_ext}"
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
            # Get any existing caption
            caption = None
            for ext in ['.txt', '.caption', '.json', '.cls']:
                ext_key = ext[1:]  # Remove the dot
                if ext_key in sample:
                    if isinstance(sample[ext_key], bytes):
                        caption = sample[ext_key].decode('utf-8')
                    else:
                        caption = sample[ext_key]
                    break
            
            # Process black bars removal
            preprocessed_path = temp_dir / f"preprocessed.{video_ext}"
            was_cropped = await self._remove_black_bars(video_path, preprocessed_path)
            
            # Use preprocessed video if cropping was done, otherwise use original
            process_path = preprocessed_path if was_cropped else video_path
            
            # Detect scenes if splitting is enabled
            if enable_splitting:
                video = open_video(str(process_path))
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector())
                
                # Use asyncio to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: scene_manager.detect_scenes(video, show_progress=False)
                )
                
                scenes = scene_manager.get_scene_list()
            else:
                scenes = []
            
            num_scenes = len(scenes)
            self._scene_counts[key] = num_scenes
            
            # Create output shard for processed content
            processed_dir = self.wds_manager.get_shard_directory("processed")
            writer = ShardWriter(processed_dir, f"processed_{uuid.uuid4().hex[:8]}")
            
            try:
                if not scenes:
                    # Single scene, just add the processed video
                    processed_key = f"{key}_001"
                    
                    # Read the processed video
                    video_content = process_path.read_bytes()
                    
                    # Add to processed shard
                    writer.add_sample(processed_key, "video", video_ext, video_content)
                    
                    # Add caption if available
                    if caption:
                        writer.add_sample(processed_key, "caption", "txt", caption.encode('utf-8'))
                    
                    # Add FPS info if caption exists
                    if caption:
                        fps_info = await loop.run_in_executor(None, get_video_fps, process_path)
                        if fps_info and not any(f"FPS, " in line for line in caption.split('\n')):
                            caption = f"{fps_info}{caption}"
                            writer.add_sample(processed_key, "caption", "txt", caption.encode('utf-8'))
                else:
                    # Multiple scenes, split the video
                    output_template = str(temp_dir / f"scene_$SCENE_NUMBER.{video_ext}")
                    
                    # Split video into scenes
                    await loop.run_in_executor(
                        None,
                        lambda: split_video_ffmpeg(
                            str(process_path),
                            scenes,
                            output_file_template=output_template,
                            show_progress=False
                        )
                    )
                    
                    # Process each scene
                    for i in range(1, num_scenes + 1):
                        scene_path = temp_dir / f"scene_{i:03d}.{video_ext}"
                        if not scene_path.exists():
                            continue
                        
                        # Create key for this scene
                        scene_key = f"{key}_{i:03d}"
                        
                        # Read the scene video
                        scene_content = scene_path.read_bytes()
                        
                        # Add to processed shard
                        writer.add_sample(scene_key, "video", video_ext, scene_content)
                        
                        # Add caption if available
                        if caption:
                            # Add parent caption to each scene
                            scene_caption = caption
                            
                            # Add FPS info if not already present
                            fps_info = await loop.run_in_executor(None, get_video_fps, scene_path)
                            if fps_info and not any(f"FPS, " in line for line in scene_caption.split('\n')):
                                scene_caption = f"{fps_info}{scene_caption}"
                            
                            writer.add_sample(scene_key, "caption", "txt", scene_caption.encode('utf-8'))
                
                # Update status
                crop_status = " (black bars removed)" if was_cropped else ""
                self._processing_status[key] = f"{num_scenes} scenes{crop_status}"
                
                if num_scenes:
                    logger.info(f"Extracted {num_scenes} clips from {key}{crop_status}")
                else:
                    logger.info(f"Processed {key}{crop_status} (no splitting)")
                
                return num_scenes
                
            finally:
                writer.close()
    
    async def _remove_black_bars(self, input_path: Path, output_path: Path) -> bool:
        """Remove black bars from video using FFmpeg
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            
        Returns:
            bool: True if cropping was performed, False otherwise
        """
        try:
            # Detect black bars
            top, bottom, left, right = await self._detect_black_bars(input_path)
            
            # Get video dimensions using OpenCV
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # If no significant black bars detected, return False
            if top < 10 and bottom > height - 10 and \
               left < 10 and right > width - 10:
                # Just copy the file
                with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                return False
            
            # Calculate crop dimensions
            crop_height = bottom - top
            crop_width = right - left
            
            if crop_height <= 0 or crop_width <= 0:
                # Just copy the file
                with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                return False
            
            # Use FFmpeg to crop and save video
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f'crop={crop_width}:{crop_height}:{left}:{top}',
                '-c:a', 'copy',  # Copy audio stream
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            # Run ffmpeg asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for the process to complete
            stdout, stderr = await process.communicate()
            
            # Check if process was successful
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                # Just copy the file as fallback
                with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing black bars: {e}")
            # Just copy the file as fallback
            with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            return False
    
    async def _detect_black_bars(self, video_path: Path) -> Tuple[int, int, int, int]:
        """Detect black bars in video by analyzing first few frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (top, bottom, left, right) crop values
        """
        loop = asyncio.get_event_loop()
        
        # Function to run on thread pool
        def _detect():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Read first few frames to get stable detection
            frames_to_check = 5
            frames = []
            
            for _ in range(frames_to_check):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise ValueError("Could not read any frames from video")
            
            # Convert frames to grayscale and find average
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            avg_frame = np.mean(gray_frames, axis=0)
            
            # Threshold to detect black regions
            threshold = 20
            black_mask = avg_frame < threshold
            
            # Find black bars by analyzing row/column means
            row_means = np.mean(black_mask, axis=1)
            col_means = np.mean(black_mask, axis=0)
            
            # Detect edges where black bars end (using high threshold to avoid false positives)
            black_threshold = 0.95  # 95% of pixels in row/col must be black
            
            # Find top and bottom crops
            top_crop = 0
            bottom_crop = black_mask.shape[0]
            
            for i, mean in enumerate(row_means):
                if mean > black_threshold:
                    top_crop = i + 1
                else:
                    break
                    
            for i, mean in enumerate(reversed(row_means)):
                if mean > black_threshold:
                    bottom_crop = black_mask.shape[0] - i - 1
                else:
                    break
            
            # Find left and right crops
            left_crop = 0
            right_crop = black_mask.shape[1]
            
            for i, mean in enumerate(col_means):
                if mean > black_threshold:
                    left_crop = i + 1
                else:
                    break
                    
            for i, mean in enumerate(reversed(col_means)):
                if mean > black_threshold:
                    right_crop = black_mask.shape[1] - i - 1
                else:
                    break
            
            return top_crop, bottom_crop, left_crop, right_crop
        
        # Run detection on thread pool
        return await loop.run_in_executor(None, _detect)
    
    async def process_legacy_video(self, video_path: Path, enable_splitting: bool) -> int:
        """Process a video file from the legacy storage system
        
        Args:
            video_path: Path to video file
            enable_splitting: Whether to split video into scenes
            
        Returns:
            Number of scenes detected
        """
        key = video_path.stem
        self._current_file = key
        self._processing_status[key] = f'Processing video "{key}"...'
        
        # Create a WebDataset sample in memory
        sample = {"__key__": key}
        
        # Read video data
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_ext = video_path.suffix.lstrip('.')
        sample[video_ext] = video_data
        
        # Check for caption file
        caption_file = video_path.with_suffix('.txt')
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                caption_data = f.read()
            sample['txt'] = caption_data
        
        # Process the video sample
        num_scenes = await self.process_video_sample(sample, enable_splitting)
        
        # Delete original video if processing was successful and we detected scenes
        if num_scenes > 0:
            video_path.unlink(missing_ok=True)
            if caption_file.exists():
                caption_file.unlink(missing_ok=True)
        
        return num_scenes
    
    async def start_processing(self, enable_splitting: bool = True) -> None:
        """Start processing all unprocessed videos in raw shards
        
        Args:
            enable_splitting: Whether to split videos into scenes
        """
        if self.processing:
            return
        
        self.processing = True
        try:
            # Get all raw shards
            raw_shards = self.wds_manager.list_shards("raw")
            
            for shard_path in raw_shards:
                if not self.processing:
                    break  # Exit if stopped
                
                # Create dataset for this shard
                url = f"file:{shard_path}"
                dataset = wds.WebDataset(url)
                
                # Process each sample in the shard
                for sample in dataset:
                    if not self.processing:
                        break  # Exit if stopped
                    
                    # Check if this is a video sample
                    is_video = False
                    for ext in ['.mp4', '.webm']:
                        ext_key = ext[1:]  # Remove the dot
                        if ext_key in sample:
                            is_video = True
                            break
                    
                    if is_video:
                        # Process the video sample
                        await self.process_video_sample(sample, enable_splitting)
            
            # Also process videos from legacy storage
            for video_path in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
                if not self.processing:
                    break  # Exit if stopped
                
                await self.process_legacy_video(video_path, enable_splitting)
        
        finally:
            self.processing = False
            self._current_file = None
    
    async def process_video(self, key: str, enable_splitting: bool = True) -> int:
        """Process a specific video by key
        
        Args:
            key: Key of video to process
            enable_splitting: Whether to split video into scenes
            
        Returns:
            Number of scenes detected
        """
        # Check if it's a file in the legacy system
        legacy_path = VIDEOS_TO_SPLIT_PATH / key
        if legacy_path.exists() and legacy_path.is_file():
            return await self.process_legacy_video(legacy_path, enable_splitting)
        
        # Find the sample in raw shards
        raw_shards = self.wds_manager.list_shards("raw")
        
        for shard_path in raw_shards:
            # Create dataset for this shard
            url = f"file:{shard_path}"
            dataset = wds.WebDataset(url)
            
            # Look for sample with matching key
            for sample in dataset:
                if sample.get("__key__") == key:
                    return await self.process_video_sample(sample, enable_splitting)
        
        self._processing_status[key] = "Video not found"
        return 0
    
    def export_to_staging(self, output_dir: Path = STAGING_PATH) -> Tuple[int, int]:
        """Export processed videos to staging directory for captioning
        
        Args:
            output_dir: Directory to export files to
        
        Returns:
            Tuple of (exported_videos, exported_images)
        """
        processed_shards = self.wds_manager.list_shards("processed")
        
        exported_videos = 0
        exported_images = 0
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for shard_path in processed_shards:
                # Create dataset for this shard
                url = f"file:{shard_path}"
                dataset = wds.WebDataset(url)
                
                # Process each sample
                for sample in dataset:
                    sample_key = sample.get("__key__")
                    if not sample_key:
                        continue
                    
                    # Determine if this is a video or image
                    is_video = any(ext in sample for ext in ["mp4", "webm"])
                    is_image = any(ext in sample for ext in ["jpg", "jpeg", "png", "webp"])
                    
                    # Skip if not media
                    if not (is_video or is_image):
                        continue
                    
                    # Get caption if available
                    caption = None
                    if "txt" in sample:
                        caption = sample["txt"]
                        if isinstance(caption, bytes):
                            caption = caption.decode('utf-8')
                    
                    # Export based on media type
                    if is_video:
                        # Find video content
                        for ext in ["mp4", "webm"]:
                            if ext in sample:
                                video_content = sample[ext]
                                # Create safe filename
                                safe_key = sample_key.replace('/', '_').replace('\\', '_')
                                output_path = output_dir / f"{safe_key}.{ext}"
                                
                                # Write video file
                                with open(output_path, 'wb') as f:
                                    f.write(video_content)
                                
                                # Write caption if available
                                if caption:
                                    caption_path = output_path.with_suffix('.txt')
                                    with open(caption_path, 'w', encoding='utf-8') as f:
                                        f.write(caption)
                                
                                exported_videos += 1
                                break
                    
                    elif is_image:
                        # Find image content
                        for ext in ["jpg", "jpeg", "png", "webp"]:
                            if ext in sample:
                                image_content = sample[ext]
                                # Create safe filename
                                safe_key = sample_key.replace('/', '_').replace('\\', '_')
                                output_path = output_dir / f"{safe_key}.{ext}"
                                
                                # Write image file
                                with open(output_path, 'wb') as f:
                                    f.write(image_content)
                                
                                # Write caption if available
                                if caption:
                                    caption_path = output_path.with_suffix('.txt')
                                    with open(caption_path, 'w', encoding='utf-8') as f:
                                        f.write(caption)
                                
                                exported_images += 1
                                break
            
            return exported_videos, exported_images
            
        except Exception as e:
            logger.error(f"Error exporting to staging: {e}")
            return exported_videos, exported_images
import os
import hashlib
import shutil
from pathlib import Path
import asyncio
import tempfile
import logging
from functools import partial
from typing import Dict, List, Optional, Tuple
import gradio as gr

from scenedetect import detect, ContentDetector, SceneManager, open_video
from scenedetect.video_splitter import split_video_ffmpeg

from config import TRAINING_PATH, STORAGE_PATH, TRAINING_VIDEOS_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, DEFAULT_PROMPT_PREFIX

from image_preprocessing import detect_black_bars
from video_preprocessing import remove_black_bars
from utils import extract_scene_info, is_video_file, is_image_file, add_prefix_to_caption

logger = logging.getLogger(__name__)

class SplittingService:
    def __init__(self):
        # Track processing status
        self.processing = False
        self._current_file: Optional[str] = None
        self._scene_counts: Dict[str, int] = {}
        self._processing_status: Dict[str, str] = {}

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def rename_with_hash(self, video_path: Path) -> Tuple[Path, str]:
        """Rename video and caption files using hash
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (new video path, hash)
        """
        # Compute hash
        file_hash = self.compute_file_hash(video_path)
        
        # Rename video file
        new_video_path = video_path.parent / f"{file_hash}{video_path.suffix}"
        video_path.rename(new_video_path)
        
        # Rename caption file if exists
        caption_path = video_path.with_suffix('.txt')
        if caption_path.exists():
            new_caption_path = caption_path.parent / f"{file_hash}.txt"
            caption_path.rename(new_caption_path)
            
        return new_video_path, file_hash

    async def process_video(self, video_path: Path, enable_splitting: bool) -> int:
        """Process a single video file to detect and split scenes"""
        try:
            self._processing_status[video_path.name] = f'Processing video "{video_path.name}"...'
            
            parent_caption_path = video_path.with_suffix('.txt')
            # Create output path for split videos
            base_name, _ = extract_scene_info(video_path.name)
            # Create temporary directory for preprocessed video
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / f"preprocessed_{video_path.name}"
                
                # Try to remove black bars
                was_cropped = await asyncio.get_event_loop().run_in_executor(
                    None,
                    remove_black_bars,
                    video_path,
                    temp_path
                )
                
                # Use preprocessed video if cropping was done, otherwise use original
                process_path = temp_path if was_cropped else video_path
                
                # Detect scenes if splitting is enabled
                if enable_splitting:
                    video = open_video(str(process_path))
                    scene_manager = SceneManager()
                    scene_manager.add_detector(ContentDetector())
                    scene_manager.detect_scenes(video, show_progress=False)
                    scenes = scene_manager.get_scene_list()
                else:
                    scenes = []

                num_scenes = len(scenes)
                    
            
        
                if not scenes:
                    print(f'video "{video_path.name}" is already a single-scene clip')

                    # captioning is only required if some information is missing

                    if parent_caption_path.exists():
                        # if it's a single scene with a caption, we can directly promote it to the training/ dir
                        #output_video_path = TRAINING_VIDEOS_PATH / f"{base_name}___{1:03d}.mp4"
                        # WELL ACTUALLY, NOT. The training videos dir removes a lot of thing,
                        # so it has to stay a "last resort" thing
                        output_video_path = STAGING_PATH / f"{base_name}___{1:03d}.mp4"
                        
                        shutil.copy2(process_path, output_video_path)
                        
                        shutil.copy2(parent_caption_path, output_video_path.with_suffix('.txt'))
                        parent_caption_path.unlink()
                    else:
                        # otherwise it needs to go through the normal captioning process
                        output_video_path = STAGING_PATH / f"{base_name}___{1:03d}.mp4"
                        shutil.copy2(process_path, output_video_path)


                else:
                    print(f'video "{video_path.name}" contains {num_scenes} scenes')

                    # in this scenario, there are multiple subscenes
                    # even if we have a parent caption, we must caption each of them individually
                    # the first step is to preserve the parent caption for later use
                    if parent_caption_path.exists():
                        output_caption_path = STAGING_PATH / f"{base_name}.txt"
                        shutil.copy2(parent_caption_path, output_caption_path)
                        parent_caption_path.unlink()


                    output_template = str(STAGING_PATH / f"{base_name}___$SCENE_NUMBER.mp4")
                    
                    # Split video into scenes using the preprocessed video if it exists
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: split_video_ffmpeg(
                            str(process_path),
                            scenes,
                            output_file_template=output_template,
                            show_progress=False
                        )
                    )

                # Update scene count and status
                crop_status = " (black bars removed)" if was_cropped else ""
                self._scene_counts[video_path.name] = num_scenes
                self._processing_status[video_path.name] = f"{num_scenes} scenes{crop_status}"
                
                # Delete original video
                video_path.unlink()
                
                if num_scenes:
                    gr.Info(f"Extracted {num_scenes} clips from {video_path.name}{crop_status}")
                else:
                    gr.Info(f"Imported {video_path.name}{crop_status}")
                    
                return num_scenes

        except Exception as e:
            self._scene_counts[video_path.name] = 0
            self._processing_status[video_path.name] = f"Error: {str(e)}"
            raise gr.Error(f"Error processing video {video_path}: {str(e)}")

    def get_scene_count(self, video_name: str) -> Optional[int]:
        """Get number of detected scenes for a video
        
        Returns None if video hasn't been scanned
        """
        return self._scene_counts.get(video_name)

    def get_current_file(self) -> Optional[str]:
        """Get name of file currently being processed"""
        return self._current_file

    def is_processing(self) -> bool:
        """Check if background processing is running"""
        return self.processing

    async def start_processing(self, enable_splitting: bool) -> None:
        """Start background processing of unprocessed videos"""
        if self.processing:
            return
            
        self.processing = True
        try:
            # Process each video
            for video_file in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
                self._current_file = video_file.name
                await self.process_video(video_file, enable_splitting)
                    
        finally:
            self.processing = False
            self._current_file = None

    def get_processing_status(self, video_name: str) -> str:
        """Get processing status for a video
        
        Args:
            video_name: Name of the video file
            
        Returns:
            Status string for the video
        """
        if video_name in self._processing_status:
            return self._processing_status[video_name]
        return "not processed"

    def list_unprocessed_videos(self) -> List[List[str]]:
        """List all unprocessed and processed videos with their status.
        Images will be ignored.
        
        Returns:
            List of lists containing [name, status] for each video
        """
        videos = []
        
        # Track processed videos by their base names
        processed_videos = {}
        for clip_path in STAGING_PATH.glob("*.mp4"):
            base_name = clip_path.stem.rsplit('___', 1)[0] + '.mp4'
            if base_name in processed_videos:
                processed_videos[base_name] += 1
            else:
                processed_videos[base_name] = 1
                
        # List only video files in processing queue
        for video_file in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
            if is_video_file(video_file):  # Only include video files
                status = self.get_processing_status(video_file.name)
                videos.append([video_file.name, status])
        
        # Add processed videos
        for video_name, clip_count in processed_videos.items():
            if not (VIDEOS_TO_SPLIT_PATH / video_name).exists():
                status = f"Processed ({clip_count} clips)"
                videos.append([video_name, status])

        return sorted(videos, key=lambda x: (x[1] != "Processing...", x[0].lower()))

import os
import shutil
import zipfile
import tempfile
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pytubefix import YouTube
import logging
from utils import is_image_file, is_video_file, add_prefix_to_caption
from image_preprocessing import normalize_image

from config import NORMALIZE_IMAGES_TO, TRAINING_VIDEOS_PATH, VIDEOS_TO_SPLIT_PATH, TRAINING_PATH, DEFAULT_PROMPT_PREFIX

logger = logging.getLogger(__name__)

class ImportService:
    def process_uploaded_files(self, file_paths: List[str]) -> str:
        """Process uploaded file (ZIP, MP4, or image)
        
        Args:
            file_paths: File paths to the ploaded files from Gradio
                
        Returns:
            Status message string
        """
        for file_path in file_paths:
            file_path = Path(file_path)
            try:
                original_name = file_path.name
                print("original_name = ", original_name)

                # Determine file type from name
                file_ext = file_path.suffix.lower()

                if file_ext == '.zip':
                    return self.process_zip_file(file_path)
                elif file_ext == '.mp4' or file_ext == '.webm':
                    return self.process_mp4_file(file_path, original_name)
                elif is_image_file(file_path):
                    return self.process_image_file(file_path, original_name)
                else:
                    raise gr.Error(f"Unsupported file type: {file_ext}")

            except Exception as e:
                raise gr.Error(f"Error processing file: {str(e)}")

    def process_image_file(self, file_path: Path, original_name: str) -> str:
        """Process a single image file
        
        Args:
            file_path: Path to the image
            original_name: Original filename
            
        Returns:
            Status message string
        """
        try:
            # Create a unique filename with configured extension
            stem = Path(original_name).stem
            target_path = STAGING_PATH / f"{stem}.{NORMALIZE_IMAGES_TO}"
            
            # If file already exists, add number suffix
            counter = 1
            while target_path.exists():
                target_path = STAGING_PATH / f"{stem}___{counter}.{NORMALIZE_IMAGES_TO}"
                counter += 1

            # Convert to normalized format and remove black bars
            success = normalize_image(file_path, target_path)
            
            if not success:
                raise gr.Error(f"Failed to process image: {original_name}")

            # Handle caption
            src_caption_path = file_path.with_suffix('.txt')
            if src_caption_path.exists():
                caption = src_caption_path.read_text()
                caption = add_prefix_to_caption(caption, DEFAULT_PROMPT_PREFIX)
                target_path.with_suffix('.txt').write_text(caption)

            logger.info(f"Successfully stored image: {target_path.name}")
            gr.Info(f"Successfully stored image: {target_path.name}")
            return f"Successfully stored image: {target_path.name}"

        except Exception as e:
            raise gr.Error(f"Error processing image file: {str(e)}")

    def process_zip_file(self, file_path: Path) -> str:
        """Process uploaded ZIP file containing media files
        
        Args:
            file_path: Path to the uploaded ZIP file
                
        Returns:
            Status message string
        """
        try:
            video_count = 0
            image_count = 0
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP
                extract_dir = Path(temp_dir) / "extracted"
                extract_dir.mkdir()
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Process each file
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file.startswith('._'):  # Skip Mac metadata
                            continue
                            
                        file_path = Path(root) / file
                        
                        try:
                            if is_video_file(file_path):
                                # Copy video to videos_to_split
                                target_path = VIDEOS_TO_SPLIT_PATH / file_path.name
                                counter = 1
                                while target_path.exists():
                                    target_path = VIDEOS_TO_SPLIT_PATH / f"{file_path.stem}___{counter}{file_path.suffix}"
                                    counter += 1
                                shutil.copy2(file_path, target_path)
                                video_count += 1
                                
                            elif is_image_file(file_path):
                                # Convert image and save to staging
                                target_path = STAGING_PATH / f"{file_path.stem}.{NORMALIZE_IMAGES_TO}"
                                counter = 1
                                while target_path.exists():
                                    target_path = STAGING_PATH / f"{file_path.stem}___{counter}.{NORMALIZE_IMAGES_TO}"
                                    counter += 1
                                if normalize_image(file_path, target_path):
                                    image_count += 1
                                
                            # Copy associated caption file if it exists
                            txt_path = file_path.with_suffix('.txt')
                            if txt_path.exists():
                                if is_video_file(file_path):
                                    shutil.copy2(txt_path, target_path.with_suffix('.txt'))
                                elif is_image_file(file_path):
                                    shutil.copy2(txt_path, target_path.with_suffix('.txt'))
                                    
                        except Exception as e:
                            logger.error(f"Error processing {file_path.name}: {str(e)}")
                            continue

            # Generate status message
            parts = []
            if video_count > 0:
                parts.append(f"{video_count} videos")
            if image_count > 0:
                parts.append(f"{image_count} images")
                
            if not parts:
                return "No supported media files found in ZIP"
                
            status = f"Successfully stored {' and '.join(parts)}"
            gr.Info(status)
            return status
            
        except Exception as e:
            raise gr.Error(f"Error processing ZIP: {str(e)}")

    def process_mp4_file(self, file_path: Path, original_name: str) -> str:
        """Process a single video file
        
        Args:
            file_path: Path to the file
            original_name: Original filename
            
        Returns:
            Status message string
        """
        try:
            # Create a unique filename
            target_path = VIDEOS_TO_SPLIT_PATH / original_name
            
            # If file already exists, add number suffix
            counter = 1
            while target_path.exists():
                stem = Path(original_name).stem
                target_path = VIDEOS_TO_SPLIT_PATH / f"{stem}___{counter}.mp4"
                counter += 1

            # Copy the file to the target location
            shutil.copy2(file_path, target_path)

            gr.Info(f"Successfully stored video: {target_path.name}")
            return f"Successfully stored video: {target_path.name}"

        except Exception as e:
            raise gr.Error(f"Error processing video file: {str(e)}")

    def download_youtube_video(self, url: str, progress=None) -> Dict:
        """Download a video from YouTube
        
        Args:
            url: YouTube video URL
            progress: Optional Gradio progress indicator
            
        Returns:
            Dict with status message and error (if any)
        """
        try:
            # Extract video ID and create YouTube object
            yt = YouTube(url, on_progress_callback=lambda stream, chunk, bytes_remaining: 
                progress((1 - bytes_remaining / stream.filesize), desc="Downloading...")
                if progress else None)
            
            video_id = yt.video_id
            output_path = VIDEOS_TO_SPLIT_PATH / f"{video_id}.mp4"
            
            # Download highest quality progressive MP4
            if progress:
                print("Getting video streams...")
                progress(0, desc="Getting video streams...")
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not video:
                print("Could not find a compatible video format")
                gr.Error("Could not find a compatible video format")
                return "Could not find a compatible video format"
            
            # Download the video
            if progress:
                print("Starting YouTube video download...")
                progress(0, desc="Starting download...")
            
            video.download(output_path=str(VIDEOS_TO_SPLIT_PATH), filename=f"{video_id}.mp4")
            
            # Update UI
            if progress:
                print("YouTube video download complete!")
                gr.Info("YouTube video download complete!")
                progress(1, desc="Download complete!")
            return f"Successfully downloaded video: {yt.title}"
            
        except Exception as e:
            print(e)
            gr.Error(f"Error downloading video: {str(e)}")
            return f"Error downloading video: {str(e)}"
"""
File upload handler for Video Model Studio.
Processes uploaded files including videos, images, ZIPs, and WebDataset archives.
"""

import os
import shutil
import zipfile
import tarfile
import tempfile
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import traceback

from vms.config import NORMALIZE_IMAGES_TO, TRAINING_VIDEOS_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, DEFAULT_PROMPT_PREFIX
from vms.utils import normalize_image, is_image_file, is_video_file, add_prefix_to_caption, webdataset_handler

logger = logging.getLogger(__name__)

class FileUploadHandler:
    """Handles processing of uploaded files"""
    
    def process_uploaded_files(self, file_paths: List[str], enable_splitting: bool) -> str:
        """Process uploaded file (ZIP, TAR, MP4, or image)
        
        Args:
            file_paths: File paths to the uploaded files from Gradio
            enable_splitting: Whether to enable automatic video splitting
                
        Returns:
            Status message string
        """
        print(f"process_uploaded_files called with enable_splitting={enable_splitting} and file_paths = {str(file_paths)}")
        if not file_paths or len(file_paths) == 0:
            logger.warning("No files provided to process_uploaded_files")
            return "No files provided"
        
        for file_path in file_paths:
            print(f" - {str(file_path)}")
            file_path = Path(file_path)
            try:
                original_name = file_path.name
                logger.info(f"Processing uploaded file: {original_name}")

                # Determine file type from name
                file_ext = file_path.suffix.lower()

                if file_ext == '.zip':
                    return self.process_zip_file(file_path, enable_splitting)
                elif file_ext == '.tar':
                    return self.process_tar_file(file_path, enable_splitting)
                elif file_ext == '.mp4' or file_ext == '.webm':
                    return self.process_mp4_file(file_path, original_name, enable_splitting)
                elif is_image_file(file_path):
                    return self.process_image_file(file_path, original_name)
                else:
                    logger.error(f"Unsupported file type: {file_ext}")
                    raise gr.Error(f"Unsupported file type: {file_ext}")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
                raise gr.Error(f"Error processing file: {str(e)}")

    def process_zip_file(self, file_path: Path, enable_splitting: bool) -> str:
        """Process uploaded ZIP file containing media files or WebDataset tar files
        
        Args:
            file_path: Path to the uploaded ZIP file
            enable_splitting: Whether to enable automatic video splitting
                
        Returns:
            Status message string
        """
        try:
            video_count = 0
            image_count = 0
            tar_count = 0
            
            logger.info(f"Processing ZIP file: {file_path}")
            
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
                            # Check if it's a WebDataset tar file
                            if file.lower().endswith('.tar'):
                                logger.info(f"Processing WebDataset archive from ZIP: {file}")
                                # Process WebDataset shard
                                vid_count, img_count = webdataset_handler.process_webdataset_shard(
                                    file_path, VIDEOS_TO_SPLIT_PATH if enable_splitting else STAGING_PATH, STAGING_PATH
                                )
                                video_count += vid_count
                                image_count += img_count
                                tar_count += 1
                            elif is_video_file(file_path):
                                # Choose target directory based on auto-splitting setting
                                target_dir = VIDEOS_TO_SPLIT_PATH if enable_splitting else STAGING_PATH
                                target_path = target_dir / file_path.name
                                counter = 1
                                while target_path.exists():
                                    target_path = target_dir / f"{file_path.stem}___{counter}{file_path.suffix}"
                                    counter += 1
                                shutil.copy2(file_path, target_path)
                                logger.info(f"Extracted video from ZIP: {file} -> {target_path.name}")
                                video_count += 1
                                
                            elif is_image_file(file_path):
                                # Convert image and save to staging
                                target_path = STAGING_PATH / f"{file_path.stem}.{NORMALIZE_IMAGES_TO}"
                                counter = 1
                                while target_path.exists():
                                    target_path = STAGING_PATH / f"{file_path.stem}___{counter}.{NORMALIZE_IMAGES_TO}"
                                    counter += 1
                                if normalize_image(file_path, target_path):
                                    logger.info(f"Extracted image from ZIP: {file} -> {target_path.name}")
                                    image_count += 1
                                
                            # Copy associated caption file if it exists
                            txt_path = file_path.with_suffix('.txt')
                            if txt_path.exists() and not file.lower().endswith('.tar'):
                                if is_video_file(file_path):
                                    shutil.copy2(txt_path, target_path.with_suffix('.txt'))
                                    logger.info(f"Copied caption file for {file}")
                                elif is_image_file(file_path):
                                    caption = txt_path.read_text()
                                    caption = add_prefix_to_caption(caption, DEFAULT_PROMPT_PREFIX)
                                    target_path.with_suffix('.txt').write_text(caption)
                                    logger.info(f"Processed caption for {file}")
                                    
                        except Exception as e:
                            logger.error(f"Error processing {file_path.name} from ZIP: {str(e)}", exc_info=True)
                            continue

            # Generate status message
            parts = []
            if tar_count > 0:
                parts.append(f"{tar_count} WebDataset shard{'s' if tar_count != 1 else ''}")
            if video_count > 0:
                parts.append(f"{video_count} video{'s' if video_count != 1 else ''}")
            if image_count > 0:
                parts.append(f"{image_count} image{'s' if image_count != 1 else ''}")
                
            if not parts:
                logger.warning("No supported media files found in ZIP")
                return "No supported media files found in ZIP"
                
            status = f"Successfully stored {', '.join(parts)}"
            logger.info(status)
            gr.Info(status)
            return status
            
        except Exception as e:
            logger.error(f"Error processing ZIP: {str(e)}", exc_info=True)
            raise gr.Error(f"Error processing ZIP: {str(e)}")

    def process_tar_file(self, file_path: Path, enable_splitting: bool) -> str:
        """Process a WebDataset tar file
        
        Args:
            file_path: Path to the uploaded tar file
            enable_splitting: Whether to enable automatic video splitting
                
        Returns:
            Status message string
        """
        try:
            logger.info(f"Processing WebDataset TAR file: {file_path}")
            video_count, image_count = webdataset_handler.process_webdataset_shard(
                file_path, VIDEOS_TO_SPLIT_PATH if enable_splitting else STAGING_PATH, STAGING_PATH
            )
            
            # Generate status message
            parts = []
            if video_count > 0:
                parts.append(f"{video_count} video{'s' if video_count != 1 else ''}")
            if image_count > 0:
                parts.append(f"{image_count} image{'s' if image_count != 1 else ''}")
                
            if not parts:
                logger.warning("No supported media files found in WebDataset")
                return "No supported media files found in WebDataset"
                
            status = f"Successfully extracted {' and '.join(parts)} from WebDataset"
            logger.info(status)
            gr.Info(status)
            return status
            
        except Exception as e:
            logger.error(f"Error processing WebDataset tar file: {str(e)}", exc_info=True)
            raise gr.Error(f"Error processing WebDataset tar file: {str(e)}")

    def process_mp4_file(self, file_path: Path, original_name: str, enable_splitting: bool) -> str:
        """Process a single video file
        
        Args:
            file_path: Path to the file
            original_name: Original filename
            enable_splitting: Whether to enable automatic video splitting
            
        Returns:
            Status message string
        """
        print(f"process_mp4_file(self, file_path={str(file_path)}, original_name={str(original_name)}, enable_splitting={enable_splitting})")
        try:
            # Choose target directory based on auto-splitting setting
            target_dir = VIDEOS_TO_SPLIT_PATH if enable_splitting else STAGING_PATH
            print(f"target_dir = {target_dir}")
            # Create a unique filename
            target_path = target_dir / original_name
            
            # If file already exists, add number suffix
            counter = 1
            while target_path.exists():
                stem = Path(original_name).stem
                target_path = target_dir / f"{stem}___{counter}.mp4"
                counter += 1

            logger.info(f"Processing video file: {original_name} -> {target_path}")
            
            # Copy the file to the target location
            shutil.copy2(file_path, target_path)

            logger.info(f"Successfully stored video: {target_path.name}")
            gr.Info(f"Successfully stored video: {target_path.name}")
            return f"Successfully stored video: {target_path.name}"

        except Exception as e:
            logger.error(f"Error processing video file: {str(e)}", exc_info=True)
            raise gr.Error(f"Error processing video file: {str(e)}")
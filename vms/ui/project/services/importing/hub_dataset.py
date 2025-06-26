"""
Hugging Face Hub dataset browser for Video Model Studio.
Handles searching, viewing, and downloading datasets from the Hub.
"""

import os
import shutil
import tempfile
import asyncio
import logging
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, Callable

from huggingface_hub import (
    HfApi, 
    hf_hub_download, 
    snapshot_download, 
    list_datasets
)

from vms.config import NORMALIZE_IMAGES_TO, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, DEFAULT_PROMPT_PREFIX
from vms.utils import normalize_image, is_image_file, is_video_file, add_prefix_to_caption, webdataset_handler

logger = logging.getLogger(__name__)

class HubDatasetBrowser:
    """Handles interactions with Hugging Face Hub datasets"""
    
    def __init__(self, hf_api: HfApi):
        """Initialize with HfApi instance
        
        Args:
            hf_api: Hugging Face Hub API instance
        """
        self.hf_api = hf_api
    
    def search_datasets(self, query: str) -> List[List[str]]:
        """Search for datasets on the Hugging Face Hub
        
        Args:
            query: Search query string
            
        Returns:
            List of datasets matching the query [id, title, downloads]
            Note: We still return all columns internally, but the UI will only display the first column
        """
        try:
            # Start with some filters to find video-related datasets
            search_terms = query.strip() if query and query.strip() else "video"
            logger.info(f"Searching datasets with query: '{search_terms}'")
            
            # Fetch datasets that match the search
            datasets = list(self.hf_api.list_datasets(
                search=search_terms,
                limit=50
            ))
            
            # Format results for display
            results = []
            for ds in datasets:
                # Extract relevant information
                dataset_id = ds.id
                
                # Safely get the title with fallbacks
                card_data = getattr(ds, "card_data", None)
                title = ""
                
                if card_data is not None and isinstance(card_data, dict):
                    title = card_data.get("name", "")
                
                if not title:
                    # Use the last part of the repo ID as a fallback
                    title = dataset_id.split("/")[-1]
                
                # Safely get downloads
                downloads = getattr(ds, "downloads", 0)
                if downloads is None:
                    downloads = 0
                
                results.append([dataset_id, title, downloads])
            
            # Sort by downloads (most downloaded first)
            results.sort(key=lambda x: x[2] if x[2] is not None else 0, reverse=True)
            
            logger.info(f"Found {len(results)} datasets matching '{search_terms}'")
            return results
        
        except Exception as e:
            logger.error(f"Error searching datasets: {str(e)}", exc_info=True)
            return [[f"Error: {str(e)}", "", ""]]
            
    def get_dataset_info(self, dataset_id: str) -> Tuple[str, Dict[str, int], Dict[str, List[str]]]:
        """Get detailed information about a dataset
        
        Args:
            dataset_id: The dataset ID to get information for
            
        Returns:
            Tuple of (markdown_info, file_counts, file_groups)
            - markdown_info: Markdown formatted string with dataset information
            - file_counts: Dictionary with counts of each file type
            - file_groups: Dictionary with lists of filenames grouped by type
        """
        try:
            if not dataset_id:
                logger.warning("No dataset ID provided to get_dataset_info")
                return "No dataset selected", {}, {}
                
            logger.info(f"Getting info for dataset: {dataset_id}")
                
            # Get detailed information about the dataset
            dataset_info = self.hf_api.dataset_info(dataset_id)
            
            # Format the information for display
            info_text = f"### {dataset_info.id}\n\n"
            
            # Add description if available (with safer access)
            card_data = getattr(dataset_info, "card_data", None)
            description = ""
            
            if card_data is not None and isinstance(card_data, dict):
                description = card_data.get("description", "")
                
            if description:
                info_text += f"{description[:500]}{'...' if len(description) > 500 else ''}\n\n"
            
            # Add basic stats (with safer access)
            #downloads = getattr(dataset_info, 'downloads', None)
            #info_text += f"## Downloads: {downloads if downloads is not None else 'N/A'}\n"
            
            #last_modified = getattr(dataset_info, 'last_modified', None)
            #info_text += f"## Last modified: {last_modified if last_modified is not None else 'N/A'}\n"
            
            # Group files by type
            file_groups = {
                "video": [],
                "webdataset": []
            }
            
            siblings = getattr(dataset_info, "siblings", None) or []
            
            # Extract files by type
            for s in siblings:
                if not hasattr(s, 'rfilename'):
                    continue
                    
                filename = s.rfilename
                if filename.lower().endswith((".mp4", ".webm")):
                    file_groups["video"].append(filename)
                elif filename.lower().endswith(".tar"):
                    file_groups["webdataset"].append(filename)
            
            # Create file counts dictionary
            file_counts = {
                "video": len(file_groups["video"]),
                "webdataset": len(file_groups["webdataset"])
            }
            
            logger.info(f"Successfully retrieved info for dataset: {dataset_id}")
            return info_text, file_counts, file_groups
            
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
            return f"Error loading dataset information: {str(e)}", {}, {}
    
    async def download_file_group(
        self, 
        dataset_id: str, 
        file_type: str, 
        enable_splitting: bool,
        progress_callback: Optional[Callable] = None,
        custom_prompt_prefix: str = None
    ) -> str:
        """Download all files of a specific type from the dataset
        
        Args:
            dataset_id: The dataset ID
            file_type: Either "video" or "webdataset"
            enable_splitting: Whether to enable automatic video splitting
            progress_callback: Optional callback for progress updates
            
        Returns:
            Status message
        """
        try:
            # Get dataset info to retrieve file list
            _, _, file_groups = self.get_dataset_info(dataset_id)
            
            # Get the list of files for the specified type
            files = file_groups.get(file_type, [])
            
            if not files:
                return f"No {file_type} files found in the dataset"
            
            logger.info(f"Downloading {len(files)} {file_type} files from dataset {dataset_id}")
            gr.Info(f"Starting download of {len(files)} {file_type} files from {dataset_id}")
            
            # Initialize progress if callback provided
            if progress_callback:
                progress_callback(0, desc=f"Starting download of {len(files)} {file_type} files", total=len(files))
            
            # Track counts for status message
            video_count = 0
            image_count = 0
            
            # Create a temporary directory for downloads
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Process all files of the requested type
                for i, filename in enumerate(files):
                    try:
                        # Update progress
                        if progress_callback:
                            progress_callback(
                                i, 
                                desc=f"Downloading file {i+1}/{len(files)}: {Path(filename).name}",
                                total=len(files)
                            )
                        
                        # Download the file
                        file_path = hf_hub_download(
                            repo_id=dataset_id,
                            filename=filename,
                            repo_type="dataset",
                            local_dir=temp_path
                        )
                        
                        file_path = Path(file_path)
                        logger.info(f"Downloaded file to {file_path}")
                        #gr.Info(f"Downloaded {file_path.name} ({i+1}/{len(files)})")
                        
                        # Process based on file type
                        if file_type == "video":
                            # Choose target directory based on auto-splitting setting
                            target_dir = VIDEOS_TO_SPLIT_PATH if enable_splitting else STAGING_PATH
                            target_path = target_dir / file_path.name
                            
                            # Make sure filename is unique
                            counter = 1
                            while target_path.exists():
                                stem = Path(file_path.name).stem
                                if "___" in stem:
                                    base_stem = stem.split("___")[0]
                                else:
                                    base_stem = stem
                                target_path = target_dir / f"{base_stem}___{counter}{Path(file_path.name).suffix}"
                                counter += 1
                                
                            # Copy the video file
                            shutil.copy2(file_path, target_path)
                            logger.info(f"Processed video: {file_path.name} -> {target_path.name}")
                            
                            # Try to download caption if it exists
                            try:
                                txt_filename = f"{Path(filename).stem}.txt"
                                for possible_path in [
                                    Path(filename).with_suffix('.txt').as_posix(),
                                    (Path(filename).parent / txt_filename).as_posix(),
                                ]:
                                    try:
                                        txt_path = hf_hub_download(
                                            repo_id=dataset_id,
                                            filename=possible_path,
                                            repo_type="dataset",
                                            local_dir=temp_path
                                        )
                                        shutil.copy2(txt_path, target_path.with_suffix('.txt'))
                                        logger.info(f"Copied caption for {file_path.name}")
                                        break
                                    except Exception:
                                        # Caption file doesn't exist at this path, try next
                                        pass
                            except Exception as e:
                                logger.warning(f"Error trying to download caption: {e}")
                            
                            video_count += 1
                            
                        elif file_type == "webdataset":
                            # Process the WebDataset archive
                            try:
                                logger.info(f"Processing WebDataset file: {file_path}")
                                vid_count, img_count = webdataset_handler.process_webdataset_shard(
                                    file_path, VIDEOS_TO_SPLIT_PATH, STAGING_PATH
                                )
                                video_count += vid_count
                                image_count += img_count
                            except Exception as e:
                                logger.error(f"Error processing WebDataset file {file_path}: {str(e)}", exc_info=True)
                    
                    except Exception as e:
                        logger.warning(f"Error processing file {filename}: {e}")
                
                # Update progress to complete
                if progress_callback:
                    progress_callback(len(files), desc="Download complete", total=len(files))
                
                # Generate status message
                if file_type == "video":
                    status_msg = f"Successfully imported {video_count} videos from dataset {dataset_id}"
                elif file_type == "webdataset":
                    parts = []
                    if video_count > 0:
                        parts.append(f"{video_count} video{'s' if video_count != 1 else ''}")
                    if image_count > 0:
                        parts.append(f"{image_count} image{'s' if image_count != 1 else ''}")
                        
                    if parts:
                        status_msg = f"Successfully imported {' and '.join(parts)} from WebDataset archives"
                    else:
                        status_msg = f"No media was found in the WebDataset archives"
                else:
                    status_msg = f"Unknown file type: {file_type}"
                
                # Final notification
                logger.info(f"✅ Download complete! {status_msg}")
                # This info message will appear as a toast notification
                gr.Info(f"✅ Download complete! {status_msg}")
                
                return status_msg
                
        except Exception as e:
            error_msg = f"Error downloading {file_type} files: {str(e)}"
            logger.error(error_msg, exc_info=True)
            gr.Error(error_msg)
            return error_msg
    
    async def download_dataset(
        self, 
        dataset_id: str, 
        enable_splitting: bool,
        progress_callback: Optional[Callable] = None,
        custom_prompt_prefix: str = None
    ) -> Tuple[str, str]:
        """Download a dataset and process its video/image content
        
        Args:
            dataset_id: The dataset ID to download
            enable_splitting: Whether to enable automatic video splitting
            progress_callback: Optional callback for progress tracking
            
        Returns:
            Tuple of (loading_msg, status_msg)
        """
        if not dataset_id:
            logger.warning("No dataset ID provided for download")
            return "No dataset selected", "Please select a dataset first"
        
        try:
            logger.info(f"Starting download of dataset: {dataset_id}")
            loading_msg = f"## Downloading dataset: {dataset_id}\n\nThis may take some time depending on the dataset size..."
            status_msg = f"Downloading dataset: {dataset_id}..."
            
            # Get dataset info to check for available files
            dataset_info = self.hf_api.dataset_info(dataset_id)
            
            # Check if there are video files or WebDataset files
            video_files = []
            tar_files = []
            
            siblings = getattr(dataset_info, "siblings", None) or []
            if siblings:
                video_files = [s.rfilename for s in siblings if hasattr(s, 'rfilename') and s.rfilename.lower().endswith((".mp4", ".webm"))]
                tar_files = [s.rfilename for s in siblings if hasattr(s, 'rfilename') and s.rfilename.lower().endswith(".tar")]
            
            # Initialize progress tracking
            total_files = len(video_files) + len(tar_files)
            if progress_callback:
                progress_callback(0, desc=f"Starting download of dataset: {dataset_id}", total=total_files)
            
            # Create a temporary directory for downloads
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                files_processed = 0
                
                # If we have video files, download them individually
                if video_files:
                    loading_msg = f"{loading_msg}\n\nDownloading {len(video_files)} video files..."
                    logger.info(f"Downloading {len(video_files)} video files from {dataset_id}")
                    
                    for i, video_file in enumerate(video_files):
                        # Update progress
                        if progress_callback:
                            progress_callback(
                                files_processed, 
                                desc=f"Downloading video {i+1}/{len(video_files)}: {Path(video_file).name}",
                                total=total_files
                            )
                            
                        # Download the video file
                        try:
                            file_path = hf_hub_download(
                                repo_id=dataset_id,
                                filename=video_file,
                                repo_type="dataset",
                                local_dir=temp_path
                            )
                            
                            # Look for associated caption file
                            txt_filename = f"{Path(video_file).stem}.txt"
                            txt_path = None
                            for possible_path in [
                                Path(video_file).with_suffix('.txt').as_posix(),
                                (Path(video_file).parent / txt_filename).as_posix(),
                            ]:
                                try:
                                    txt_path = hf_hub_download(
                                        repo_id=dataset_id,
                                        filename=possible_path,
                                        repo_type="dataset",
                                        local_dir=temp_path
                                    )
                                    logger.info(f"Found caption file for {video_file}: {possible_path}")
                                    break
                                except Exception as e:
                                    # Caption file doesn't exist at this path, try next
                                    logger.debug(f"No caption at {possible_path}: {str(e)}")
                                    pass
                                
                            status_msg = f"Downloaded video {i+1}/{len(video_files)} from {dataset_id}"
                            logger.info(status_msg)
                            files_processed += 1
                        except Exception as e:
                            logger.warning(f"Error downloading {video_file}: {e}")
                
                # If we have tar files, download them
                if tar_files:
                    loading_msg = f"{loading_msg}\n\nDownloading {len(tar_files)} WebDataset files..."
                    logger.info(f"Downloading {len(tar_files)} WebDataset files from {dataset_id}")
                    
                    for i, tar_file in enumerate(tar_files):
                        # Update progress
                        if progress_callback:
                            progress_callback(
                                files_processed, 
                                desc=f"Downloading WebDataset {i+1}/{len(tar_files)}: {Path(tar_file).name}",
                                total=total_files
                            )
                            
                        try:
                            file_path = hf_hub_download(
                                repo_id=dataset_id,
                                filename=tar_file,
                                repo_type="dataset",
                                local_dir=temp_path
                            )
                            status_msg = f"Downloaded WebDataset {i+1}/{len(tar_files)} from {dataset_id}"
                            logger.info(status_msg)
                            files_processed += 1
                        except Exception as e:
                            logger.warning(f"Error downloading {tar_file}: {e}")
                
                # If no specific files were found, try downloading the entire repo
                if not video_files and not tar_files:
                    loading_msg = f"{loading_msg}\n\nDownloading entire dataset repository..."
                    logger.info(f"No specific media files found, downloading entire repository for {dataset_id}")
                    
                    if progress_callback:
                        progress_callback(0, desc=f"Downloading entire repository for {dataset_id}", total=1)
                    
                    try:
                        snapshot_download(
                            repo_id=dataset_id,
                            repo_type="dataset",
                            local_dir=temp_path
                        )
                        status_msg = f"Downloaded entire repository for {dataset_id}"
                        logger.info(status_msg)
                        
                        if progress_callback:
                            progress_callback(1, desc="Repository download complete", total=1)
                    except Exception as e:
                        logger.error(f"Error downloading dataset snapshot: {e}", exc_info=True)
                        return loading_msg, f"Error downloading dataset: {str(e)}"
                
                # Process the downloaded files
                loading_msg = f"{loading_msg}\n\nProcessing downloaded files..."
                logger.info(f"Processing downloaded files from {dataset_id}")
                
                if progress_callback:
                    progress_callback(0, desc="Processing downloaded files", total=100)
                
                # Count imported files
                video_count = 0
                image_count = 0
                tar_count = 0
                
                # Process function for the event loop
                async def process_files():
                    nonlocal video_count, image_count, tar_count
                    
                    # Get total number of files to process
                    file_count = 0
                    for root, _, files in os.walk(temp_path):
                        file_count += len(files)
                    
                    processed = 0
                    
                    # Process all files in the temp directory
                    for root, _, files in os.walk(temp_path):
                        for file in files:
                            file_path = Path(root) / file
                            
                            # Update progress (every 5 files to avoid too many updates)
                            if progress_callback and processed % 5 == 0:
                                if file_count > 0:
                                    progress_percent = int((processed / file_count) * 100)
                                    progress_callback(
                                        progress_percent, 
                                        desc=f"Processing files: {processed}/{file_count}",
                                        total=100
                                    )
                            
                            # Process videos
                            if file.lower().endswith((".mp4", ".webm")):
                                # Choose target path based on auto-splitting setting
                                target_dir = VIDEOS_TO_SPLIT_PATH if enable_splitting else STAGING_PATH
                                target_path = target_dir / file_path.name
                                
                                # Make sure filename is unique
                                counter = 1
                                while target_path.exists():
                                    stem = Path(file_path.name).stem
                                    if "___" in stem:
                                        base_stem = stem.split("___")[0]
                                    else:
                                        base_stem = stem
                                    target_path = target_dir / f"{base_stem}___{counter}{Path(file_path.name).suffix}"
                                    counter += 1
                                    
                                # Copy the video file
                                shutil.copy2(file_path, target_path)
                                logger.info(f"Processed video from dataset: {file_path.name} -> {target_path.name}")
                                
                                # Copy associated caption file if it exists
                                txt_path = file_path.with_suffix('.txt')
                                if txt_path.exists():
                                    shutil.copy2(txt_path, target_path.with_suffix('.txt'))
                                    logger.info(f"Copied caption for {file_path.name}")
                                    
                                video_count += 1
                                
                            # Process images
                            elif is_image_file(file_path):
                                target_path = STAGING_PATH / f"{file_path.stem}.{NORMALIZE_IMAGES_TO}"
                                
                                counter = 1
                                while target_path.exists():
                                    target_path = STAGING_PATH / f"{file_path.stem}___{counter}.{NORMALIZE_IMAGES_TO}"
                                    counter += 1
                                    
                                if normalize_image(file_path, target_path):
                                    logger.info(f"Processed image from dataset: {file_path.name} -> {target_path.name}")
                                    
                                    # Copy caption if available
                                    txt_path = file_path.with_suffix('.txt')
                                    if txt_path.exists():
                                        caption = txt_path.read_text()
                                        caption = add_prefix_to_caption(caption, custom_prompt_prefix or DEFAULT_PROMPT_PREFIX)
                                        target_path.with_suffix('.txt').write_text(caption)
                                        logger.info(f"Processed caption for {file_path.name}")
                                    
                                    image_count += 1
                                
                            # Process WebDataset files
                            elif file.lower().endswith(".tar"):
                                # Process the WebDataset archive
                                try:
                                    logger.info(f"Processing WebDataset file from dataset: {file}")
                                    vid_count, img_count = webdataset_handler.process_webdataset_shard(
                                        file_path, VIDEOS_TO_SPLIT_PATH, STAGING_PATH
                                    )
                                    tar_count += 1
                                    video_count += vid_count
                                    image_count += img_count
                                    logger.info(f"Extracted {vid_count} videos and {img_count} images from {file}")
                                except Exception as e:
                                    logger.error(f"Error processing WebDataset file {file_path}: {str(e)}", exc_info=True)
                            
                            processed += 1
                                    
                # Run the processing asynchronously
                await process_files()
                
                # Update progress to complete
                if progress_callback:
                    progress_callback(100, desc="Processing complete", total=100)
                
                # Generate final status message
                parts = []
                if video_count > 0:
                    parts.append(f"{video_count} video{'s' if video_count != 1 else ''}")
                if image_count > 0:
                    parts.append(f"{image_count} image{'s' if image_count != 1 else ''}")
                if tar_count > 0:
                    parts.append(f"{tar_count} WebDataset archive{'s' if tar_count != 1 else ''}")
                
                if parts:
                    status = f"Successfully imported {', '.join(parts)} from dataset {dataset_id}"
                    loading_msg = f"{loading_msg}\n\n✅ Success! {status}"
                    logger.info(status)
                else:
                    status = f"No supported media files found in dataset {dataset_id}"
                    loading_msg = f"{loading_msg}\n\n⚠️ {status}"
                    logger.warning(status)
                
                gr.Info(status)
                return loading_msg, status
                
        except Exception as e:
            error_msg = f"Error downloading dataset {dataset_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}", error_msg
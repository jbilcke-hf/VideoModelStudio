"""
Main Import Service for Video Model Studio.
Delegates to specialized handler classes for different import types.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import gradio as gr

from huggingface_hub import HfApi

from vms.config import HF_API_TOKEN

from vms.ui.project.services.importing.file_upload import FileUploadHandler
from vms.ui.project.services.importing.youtube import YouTubeDownloader
from vms.ui.project.services.importing.hub_dataset import HubDatasetBrowser

logger = logging.getLogger(__name__)

class ImportingService:
    """Main service class for handling imports from various sources"""
    
    def __init__(self):
        """Initialize the import service and handlers"""
        self.hf_api = HfApi(token=HF_API_TOKEN)
        self.file_handler = FileUploadHandler()
        self.youtube_handler = YouTubeDownloader()
        self.hub_browser = HubDatasetBrowser(self.hf_api)
    
    def process_uploaded_files(self, file_paths: List[str], enable_splitting: bool, custom_prompt_prefix: str = None) -> str:
        """Process uploaded file (ZIP, TAR, MP4, or image)
        
        Args:
            file_paths: File paths to the uploaded files from Gradio
            enable_splitting: Whether to enable automatic video splitting
                
        Returns:
            Status message string
        """
        print(f"process_uploaded_files(...,  enable_splitting = { enable_splitting})")
        if not file_paths or len(file_paths) == 0:
            logger.warning("No files provided to process_uploaded_files")
            return "No files provided"
        
        print(f"process_uploaded_files(..., enable_splitting = {enable_splitting:})")
        print(f"process_uploaded_files: calling self.file_handler.process_uploaded_files")
        return self.file_handler.process_uploaded_files(file_paths, enable_splitting, custom_prompt_prefix)
    
    def download_youtube_video(self, url: str, enable_splitting: bool, progress=None) -> str:
        """Download a video from YouTube
        
        Args:
            url: YouTube video URL
            enable_splitting: Whether to enable automatic video splitting
            progress: Optional Gradio progress indicator
            
        Returns:
            Status message string
        """
        return self.youtube_handler.download_video(url, enable_splitting, progress)
    
    def search_datasets(self, query: str) -> List[List[str]]:
        """Search for datasets on the Hugging Face Hub
        
        Args:
            query: Search query string
            
        Returns:
            List of datasets matching the query [id, title, downloads]
        """
        return self.hub_browser.search_datasets(query)
    
    def get_dataset_info(self, dataset_id: str) -> Tuple[str, Dict[str, int], Dict[str, List[str]]]:
        """Get detailed information about a dataset
        
        Args:
            dataset_id: The dataset ID to get information for
            
        Returns:
            Tuple of (markdown_info, file_counts, file_groups)
        """
        return self.hub_browser.get_dataset_info(dataset_id)
    
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
        return await self.hub_browser.download_dataset(dataset_id, enable_splitting, progress_callback, custom_prompt_prefix)
    
    async def download_file_group(
        self, 
        dataset_id: str, 
        file_type: str, 
        enable_splitting: bool,
        progress_callback: Optional[Callable] = None,
        custom_prompt_prefix: str = None
    ) -> str:
        """Download a group of files (videos or WebDatasets)
        
        Args:
            dataset_id: The dataset ID
            file_type: Type of file ("video" or "webdataset")
            enable_splitting: Whether to enable automatic video splitting
            progress_callback: Optional callback for progress tracking
            
        Returns:
            Status message
        """
        return await self.hub_browser.download_file_group(dataset_id, file_type, enable_splitting, progress_callback, custom_prompt_prefix)
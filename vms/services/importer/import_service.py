"""
Main Import Service for Video Model Studio.
Delegates to specialized handler classes for different import types.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import gradio as gr

from huggingface_hub import HfApi

from .file_upload import FileUploadHandler
from .youtube import YouTubeDownloader
from .hub_dataset import HubDatasetBrowser
from vms.config import HF_API_TOKEN

logger = logging.getLogger(__name__)

class ImportService:
    """Main service class for handling imports from various sources"""
    
    def __init__(self):
        """Initialize the import service and handlers"""
        self.hf_api = HfApi(token=HF_API_TOKEN)
        self.file_handler = FileUploadHandler()
        self.youtube_handler = YouTubeDownloader()
        self.hub_browser = HubDatasetBrowser(self.hf_api)
    
    def process_uploaded_files(self, file_paths: List[str]) -> str:
        """Process uploaded file (ZIP, TAR, MP4, or image)
        
        Args:
            file_paths: File paths to the uploaded files from Gradio
                
        Returns:
            Status message string
        """
        if not file_paths or len(file_paths) == 0:
            logger.warning("No files provided to process_uploaded_files")
            return "No files provided"
        
        return self.file_handler.process_uploaded_files(file_paths)
    
    def download_youtube_video(self, url: str, progress=None) -> str:
        """Download a video from YouTube
        
        Args:
            url: YouTube video URL
            progress: Optional Gradio progress indicator
            
        Returns:
            Status message string
        """
        return self.youtube_handler.download_video(url, progress)
    
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
        enable_splitting: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[str, str]:
        """Download a dataset and process its video/image content
        
        Args:
            dataset_id: The dataset ID to download
            enable_splitting: Whether to enable automatic video splitting
            progress_callback: Optional callback for progress tracking
            
        Returns:
            Tuple of (loading_msg, status_msg)
        """
        return await self.hub_browser.download_dataset(dataset_id, enable_splitting, progress_callback)
    
    async def download_file_group(
        self, 
        dataset_id: str, 
        file_type: str, 
        enable_splitting: bool = True,
        progress_callback: Optional[Callable] = None
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
        return await self.hub_browser.download_file_group(dataset_id, file_type, enable_splitting, progress_callback)
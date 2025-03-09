"""
YouTube downloader for Video Model Studio.
Handles downloading videos from YouTube URLs.
"""

import logging
import gradio as gr
from pathlib import Path
from typing import Optional, Any, Union, Callable

from pytubefix import YouTube

from vms.config import VIDEOS_TO_SPLIT_PATH

logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """Handles downloading videos from YouTube"""
    
    def download_video(self, url: str, progress: Optional[Callable] = None) -> str:
        """Download a video from YouTube
        
        Args:
            url: YouTube video URL
            progress: Optional Gradio progress indicator
            
        Returns:
            Status message string
        """
        if not url or not url.strip():
            logger.warning("No YouTube URL provided")
            return "Please enter a YouTube URL"
            
        try:
            logger.info(f"Downloading YouTube video: {url}")
            
            # Extract video ID and create YouTube object
            yt = YouTube(url, on_progress_callback=lambda stream, chunk, bytes_remaining: 
                progress((1 - bytes_remaining / stream.filesize), desc="Downloading...")
                if progress else None)
            
            video_id = yt.video_id
            output_path = VIDEOS_TO_SPLIT_PATH / f"{video_id}.mp4"
            
            # Download highest quality progressive MP4
            if progress:
                logger.debug("Getting video streams...")
                progress(0, desc="Getting video streams...")
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not video:
                logger.error("Could not find a compatible video format")
                gr.Error("Could not find a compatible video format")
                return "Could not find a compatible video format"
            
            # Download the video
            if progress:
                logger.info("Starting YouTube video download...")
                progress(0, desc="Starting download...")
            
            video.download(output_path=str(VIDEOS_TO_SPLIT_PATH), filename=f"{video_id}.mp4")
            
            # Update UI
            if progress:
                logger.info("YouTube video download complete!")
                gr.Info("YouTube video download complete!")
                progress(1, desc="Download complete!")
            return f"Successfully downloaded video: {yt.title}"
            
        except Exception as e:
            logger.error(f"Error downloading YouTube video: {str(e)}", exc_info=True)
            gr.Error(f"Error downloading video: {str(e)}")
            return f"Error downloading video: {str(e)}"
"""
Parent import tab for Video Model Studio UI that contains sub-tabs
"""

import gradio as gr
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..base_tab import BaseTab
from .upload_tab import UploadTab
from .youtube_tab import YouTubeTab
from .hub_tab import HubTab

from vms.config import (
    VIDEOS_TO_SPLIT_PATH, DEFAULT_PROMPT_PREFIX, DEFAULT_CAPTIONING_BOT_INSTRUCTIONS,
    STAGING_PATH
)

logger = logging.getLogger(__name__)

class ImportTab(BaseTab):
    """Import tab for uploading videos and images, and browsing datasets"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "import_tab"
        self.title = "1️⃣  Import"
        # Initialize sub-tabs
        self.upload_tab = UploadTab(app_state)
        self.youtube_tab = YouTubeTab(app_state)
        self.hub_tab = HubTab(app_state)
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Import tab UI components with three sub-tabs"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                gr.Markdown("## Import settings")
            
            with gr.Row():
                self.components["enable_automatic_video_split"] = gr.Checkbox(
                    label="Automatically split videos into smaller clips",
                    info="Note: a clip is a single camera shot, usually a few seconds",
                    value=True,
                    visible=True
                )
                self.components["enable_automatic_content_captioning"] = gr.Checkbox(
                    label="Automatically caption photos and videos",
                    info="Note: this uses LlaVA and takes some extra time to load and process",
                    value=False,
                    visible=True,
                )
            
            # Create tabs for different import methods
            with gr.Tabs() as import_tabs:
                # Create each sub-tab
                self.upload_tab.create(import_tabs)
                self.youtube_tab.create(import_tabs)
                self.hub_tab.create(import_tabs)
                
                # Store references to sub-tabs
                self.components["upload_tab"] = self.upload_tab
                self.components["youtube_tab"] = self.youtube_tab
                self.components["hub_tab"] = self.hub_tab
            
            with gr.Row():
                self.components["import_status"] = gr.Textbox(label="Status", interactive=False)

        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Set shared components from parent tab to sub-tabs first
        for subtab in [self.upload_tab, self.youtube_tab, self.hub_tab]:
            subtab.components["import_status"] = self.components["import_status"]
            subtab.components["enable_automatic_video_split"] = self.components["enable_automatic_video_split"]
            subtab.components["enable_automatic_content_captioning"] = self.components["enable_automatic_content_captioning"]
            
        # Then connect events for each sub-tab
        self.upload_tab.connect_events()
        self.youtube_tab.connect_events()
        self.hub_tab.connect_events()
    
    async def on_import_success(self, enable_splitting, enable_automatic_content_captioning, prompt_prefix):
        """Handle successful import of files"""
        videos = self.app.tabs["split_tab"].list_unprocessed_videos()
        
        # If scene detection isn't already running and there are videos to process,
        # and auto-splitting is enabled, start the detection
        if videos and not self.app.splitter.is_processing() and enable_splitting:
            await self.app.tabs["split_tab"].start_scene_detection(enable_splitting)
            msg = "Starting automatic scene detection..."
        else:
            # Just copy files without splitting if auto-split disabled
            for video_file in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
                await self.app.splitter.process_video(video_file, enable_splitting=False)
            msg = "Copying videos without splitting..."
        
        self.app.tabs["caption_tab"].copy_files_to_training_dir(prompt_prefix)

        # Start auto-captioning if enabled, and handle async generator properly
        if enable_automatic_content_captioning:
            # Create a background task for captioning
            asyncio.create_task(self.app.tabs["caption_tab"]._process_caption_generator(
                DEFAULT_CAPTIONING_BOT_INSTRUCTIONS,
                prompt_prefix
            ))
        
        return {
            "tabs": gr.Tabs(selected="split_tab"),
            "video_list": videos,
            "detect_status": msg
        }
        
    async def update_titles_after_import(self, enable_splitting, enable_automatic_content_captioning, prompt_prefix):
        """Handle post-import updates including titles"""
        import_result = await self.on_import_success(enable_splitting, enable_automatic_content_captioning, prompt_prefix)
        titles = self.app.update_titles()
        return (
            import_result["tabs"],
            import_result["video_list"],
            import_result["detect_status"],
            *titles
        )
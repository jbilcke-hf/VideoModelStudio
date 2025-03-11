"""
Parent import tab for Video Model Studio UI that contains sub-tabs
"""

import gradio as gr
import logging
import asyncio
import threading
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
    
    def on_import_success(self, enable_splitting, enable_automatic_content_captioning, prompt_prefix):
        """Handle successful import of files"""
        videos = self.app.tabs["split_tab"].list_unprocessed_videos()
        
        # If scene detection isn't already running and there are videos to process,
        # and auto-splitting is enabled, start the detection
        if videos and not self.app.splitting.is_processing() and enable_splitting:
            # Start the scene detection in a separate thread
            self._start_scene_detection_bg(enable_splitting)
            msg = "Starting automatic scene detection..."
        else:
            # Just copy files without splitting if auto-split disabled
            self._start_copy_files_bg(enable_splitting)
            msg = "Copying videos without splitting..."
        
        self.app.tabs["caption_tab"].copy_files_to_training_dir(prompt_prefix)

        # Start auto-captioning if enabled
        if enable_automatic_content_captioning:
            self._start_captioning_bg(DEFAULT_CAPTIONING_BOT_INSTRUCTIONS, prompt_prefix)
        
        # Return the correct tuple of values as expected by the UI
        return gr.update(selected="split_tab"), videos, msg
    
    def _start_scene_detection_bg(self, enable_splitting):
        """Start scene detection in a background thread"""
        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.app.tabs["split_tab"].start_scene_detection(enable_splitting)
                )
            except Exception as e:
                logger.error(f"Error in background scene detection: {str(e)}", exc_info=True)
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async_in_thread)
        thread.daemon = True
        thread.start()
    
    def _start_copy_files_bg(self, enable_splitting):
        """Start copying files in a background thread"""
        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def copy_files():
                    for video_file in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
                        await self.app.splitting.process_video(video_file, enable_splitting=False)
                
                loop.run_until_complete(copy_files())
            except Exception as e:
                logger.error(f"Error in background file copying: {str(e)}", exc_info=True)
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async_in_thread)
        thread.daemon = True
        thread.start()
    
    def _start_captioning_bg(self, instructions, prompt_prefix):
        """Start captioning in a background thread"""
        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.app.tabs["caption_tab"]._process_caption_generator(
                        instructions, prompt_prefix
                    )
                )
            except Exception as e:
                logger.error(f"Error in background captioning: {str(e)}", exc_info=True)
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async_in_thread)
        thread.daemon = True
        thread.start()
        
    async def update_titles_after_import(self, enable_splitting, enable_automatic_content_captioning, prompt_prefix):
        """Handle post-import updates including titles"""
        # Call the non-async version since we need to return immediately for the UI
        tabs, video_list, detect_status = self.on_import_success(
            enable_splitting, enable_automatic_content_captioning, prompt_prefix
        )
        
        # Get updated titles
        titles = self.app.update_titles()
        
        # Return all expected outputs
        return tabs, video_list, detect_status, *titles
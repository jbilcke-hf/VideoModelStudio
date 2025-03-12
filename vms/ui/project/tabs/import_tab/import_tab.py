"""
Parent import tab for Video Model Studio UI that contains sub-tabs
"""

import gradio as gr
import logging
import asyncio
import threading
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from vms.utils import BaseTab
from vms.ui.project.tabs.import_tab.upload_tab import UploadTab
from vms.ui.project.tabs.import_tab.youtube_tab import YouTubeTab
from vms.ui.project.tabs.import_tab.hub_tab import HubTab

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
        
        # Initialize sub-tabs - these should be created first
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
            
            # Create the import status textbox before creating the sub-tabs
            with gr.Row():
                self.components["import_status"] = gr.Textbox(label="Status", interactive=False)
            
            # Now create tabs for different import methods
            with gr.Tabs() as import_tabs:
                # Create each sub-tab
                self.upload_tab.create(import_tabs)
                self.youtube_tab.create(import_tabs)
                self.hub_tab.create(import_tabs)
                
                # Store references to sub-tabs
                self.components["upload_tab"] = self.upload_tab
                self.components["youtube_tab"] = self.youtube_tab
                self.components["hub_tab"] = self.hub_tab

        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Set shared components from parent tab to sub-tabs before connecting events
        for subtab in [self.upload_tab, self.youtube_tab, self.hub_tab]:
            # Ensure these components exist in the parent before sharing them
            if "import_status" in self.components:
                subtab.components["import_status"] = self.components["import_status"]
            if "enable_automatic_video_split" in self.components:
                subtab.components["enable_automatic_video_split"] = self.components["enable_automatic_video_split"]
            if "enable_automatic_content_captioning" in self.components:
                subtab.components["enable_automatic_content_captioning"] = self.components["enable_automatic_content_captioning"]
            
        # Then connect events for each sub-tab
        self.upload_tab.connect_events()
        self.youtube_tab.connect_events()
        self.hub_tab.connect_events()
    
    def on_import_success(self, enable_splitting, enable_automatic_content_captioning, prompt_prefix):
        """Handle successful import of files"""
        # If splitting is disabled, we need to directly move videos to staging
        if not enable_splitting:
            # Copy files without splitting
            self._start_copy_to_staging_bg()
            msg = "Copying videos to staging directory without splitting..."
        else:
            # Start scene detection if not already running and there are videos to process
            if not self.app.splitting.is_processing():
                # Start the scene detection in a separate thread
                self._start_scene_detection_bg(enable_splitting)
                msg = "Starting automatic scene detection..."
            else:
                msg = "Scene detection already running..."

        # Copy files to training directory
        self.app.tabs["caption_tab"].copy_files_to_training_dir(prompt_prefix)
        
        # Start auto-captioning if enabled
        if enable_automatic_content_captioning:
            self._start_captioning_bg(DEFAULT_CAPTIONING_BOT_INSTRUCTIONS, prompt_prefix)
        
        # Check if we have access to project_tabs_component for tab switching
        if hasattr(self.app, "project_tabs_component") and self.app.project_tabs_component is not None:
            # Now redirect to the caption tab instead of split tab
            return gr.update(selected="caption_tab"), msg
        else:
            # If no tabs component is available, just return the message
            logger.warning("Cannot switch tabs - project_tabs_component not available")
            return None, msg
    
    def _start_scene_detection_bg(self, enable_splitting):
        """Start scene detection in a background thread"""
        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.app.splitting.start_processing(enable_splitting)
                )
            except Exception as e:
                logger.error(f"Error in background scene detection: {str(e)}", exc_info=True)
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async_in_thread)
        thread.daemon = True
        thread.start()
    
    def _start_copy_to_staging_bg(self):
        """Start copying files directly to staging directory in a background thread"""
        def run_async_in_thread():
            try:
                # Copy all videos from videos_to_split to staging without scene detection
                for video_file in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
                    try:
                        # Ensure unique filename in staging directory
                        target_path = STAGING_PATH / video_file.name
                        counter = 1
                        
                        while target_path.exists():
                            stem = video_file.stem
                            if "___" in stem:
                                base_stem = stem.split("___")[0]
                            else:
                                base_stem = stem
                            target_path = STAGING_PATH / f"{base_stem}___{counter}{video_file.suffix}"
                            counter += 1
                        
                        # Copy the video file to staging
                        shutil.copy2(video_file, target_path)
                        logger.info(f"Copied video directly to staging: {video_file.name} -> {target_path.name}")
                        
                        # Copy caption file if it exists
                        caption_path = video_file.with_suffix('.txt')
                        if caption_path.exists():
                            shutil.copy2(caption_path, target_path.with_suffix('.txt'))
                            logger.info(f"Copied caption for {video_file.name}")
                        
                        # Remove original after successful copy
                        video_file.unlink()
                        if caption_path.exists():
                            caption_path.unlink()
                            
                        gr.Info(f"Imported {video_file.name} directly to staging")
                        
                    except Exception as e:
                        logger.error(f"Error copying {video_file.name} to staging: {str(e)}", exc_info=True)
                
            except Exception as e:
                logger.error(f"Error in background file copying to staging: {str(e)}", exc_info=True)
        
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
        tabs, status_msg = self.on_import_success(
            enable_splitting, enable_automatic_content_captioning, prompt_prefix
        )
        
        # Get updated titles
        titles = self.app.update_titles()
        
        # Return all expected outputs
        return tabs, status_msg, *titles
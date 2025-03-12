"""
Upload tab for Video Model Studio UI.
Handles manual file uploads for videos, images, and archives.
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from vms.utils import BaseTab

logger = logging.getLogger(__name__)

class UploadTab(BaseTab):
    """Upload tab for manual file uploads"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "upload_tab"
        self.title = "Manual Upload"
        # Initialize the components dictionary with None values for expected shared components
        if "components" not in self.__dict__:
            self.components = {}
        self.components["import_status"] = None
        self.components["enable_automatic_video_split"] = None
        self.components["enable_automatic_content_captioning"] = None
    
    def create(self, parent=None) -> gr.Tab:
        """Create the Upload tab UI components"""
        with gr.Tab(self.title, id=self.id) as tab:
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Manual upload of video files")
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("You can upload either:")
                        with gr.Row():
                            gr.Markdown("- A single MP4 video file")
                        with gr.Row():
                            gr.Markdown("- A ZIP archive containing multiple videos/images and optional caption files")
                        with gr.Row():
                            gr.Markdown("- A WebDataset shard (.tar file)")
                        with gr.Row():
                            gr.Markdown("- A ZIP archive containing WebDataset shards (.tar files)")
                    with gr.Column():
                        with gr.Row():
                            self.components["files"] = gr.Files(
                                label="Upload Images, Videos, ZIP or WebDataset",
                                file_types=[".jpg", ".jpeg", ".png", ".webp", ".webp", ".avif", ".heic", ".mp4", ".zip", ".tar"],
                                type="filepath"
                            )
            
            return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Check if required shared components exist before connecting events
        if not self.components.get("import_status"):
            logger.warning("import_status component is not set in UploadTab")
            return
        
        # File upload event
        upload_event = self.components["files"].upload(
            fn=lambda x: self.app.importing.process_uploaded_files(x),
            inputs=[self.components["files"]],
            outputs=[self.components["import_status"]]
        )
        
        # Only add success handler if all required components exist
        if hasattr(self.app.tabs, "import_tab") and hasattr(self.app.tabs, "caption_tab") and hasattr(self.app.tabs, "train_tab"):
            
            # Get required components for success handler
            try:
                # If the components are missing, this will raise an AttributeError
                if hasattr(self.app, "project_tabs_component"):
                    tabs_component = self.app.project_tabs_component
                else:
                    logger.warning("project_tabs_component not found in app, using None for tab switching")
                    tabs_component = None
                    
                caption_title = self.app.tabs["caption_tab"].components["caption_title"]
                train_title = self.app.tabs["train_tab"].components["train_title"]
                custom_prompt_prefix = self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
                
                # Add success handler
                upload_event.success(
                    fn=self.app.tabs["import_tab"].update_titles_after_import,
                    inputs=[
                        self.components["enable_automatic_video_split"], 
                        self.components["enable_automatic_content_captioning"], 
                        custom_prompt_prefix
                    ],
                    outputs=[
                        tabs_component,
                        self.components["import_status"],
                        caption_title,
                        train_title
                    ]
                )
            except (AttributeError, KeyError) as e:
                logger.error(f"Error connecting event handlers in UploadTab: {str(e)}")
                # Continue without the success handler
"""
Upload tab for Video Model Studio UI.
Handles manual file uploads for videos, images, and archives.
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..base_tab import BaseTab

logger = logging.getLogger(__name__)

class UploadTab(BaseTab):
    """Upload tab for manual file uploads"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "upload_tab"
        self.title = "Manual Upload"
    
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
        # File upload event
        self.components["files"].upload(
            fn=lambda x: self.app.importer.process_uploaded_files(x),
            inputs=[self.components["files"]],
            outputs=[self.components["import_status"]]  # This comes from parent tab
        ).success(
            fn=self.app.tabs["import_tab"].update_titles_after_import,
            inputs=[
                self.components["enable_automatic_video_split"], 
                self.components["enable_automatic_content_captioning"], 
                self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
            ],
            outputs=[
                self.app.tabs_component,  # Main tabs component 
                self.app.tabs["split_tab"].components["video_list"],
                self.app.tabs["split_tab"].components["detect_status"],
                self.app.tabs["split_tab"].components["split_title"],
                self.app.tabs["caption_tab"].components["caption_title"],
                self.app.tabs["train_tab"].components["train_title"]
            ]
        )
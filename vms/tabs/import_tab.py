"""
Import tab for Video Model Studio UI
"""

import gradio as gr
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_tab import BaseTab
from ..config import (
    VIDEOS_TO_SPLIT_PATH, DEFAULT_PROMPT_PREFIX, DEFAULT_CAPTIONING_BOT_INSTRUCTIONS
)

logger = logging.getLogger(__name__)

class ImportTab(BaseTab):
    """Import tab for uploading videos and images"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "import_tab"
        self.title = "1️⃣  Import"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Import tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                gr.Markdown("## Automatic splitting and captioning")
            
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
                
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Import video files")
                            gr.Markdown("You can upload either:")
                            gr.Markdown("- A single MP4 video file")
                            gr.Markdown("- A ZIP archive containing multiple videos and optional caption files")
                            gr.Markdown("For ZIP files: Create a folder containing videos (name is not important) and optional caption files with the same name (eg. `some_video.txt` for `some_video.mp4`)")
                                
                    with gr.Row():
                        self.components["files"] = gr.Files(
                            label="Upload Images, Videos or ZIP",
                            file_types=[".jpg", ".jpeg", ".png", ".webp", ".webp", ".avif", ".heic", ".mp4", ".zip"],
                            type="filepath"
                        )
       
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Import a YouTube video")
                            gr.Markdown("You can also use a YouTube video as reference, by pasting its URL here:")

                    with gr.Row():
                        self.components["youtube_url"] = gr.Textbox(
                            label="Import YouTube Video",
                            placeholder="https://www.youtube.com/watch?v=..."
                        )
                    with gr.Row():
                        self.components["youtube_download_btn"] = gr.Button("Download YouTube Video", variant="secondary")
            with gr.Row():
                self.components["import_status"] = gr.Textbox(label="Status", interactive=False)

        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # File upload event
        self.components["files"].upload(
            fn=lambda x: self.app.importer.process_uploaded_files(x),
            inputs=[self.components["files"]],
            outputs=[self.components["import_status"]]
        ).success(
            fn=self.app.update_titles_after_import,
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
        
        # YouTube download event
        self.components["youtube_download_btn"].click(
            fn=self.app.importer.download_youtube_video,
            inputs=[self.components["youtube_url"]],
            outputs=[self.components["import_status"]]
        ).success(
            fn=self.app.on_import_success,
            inputs=[
                self.components["enable_automatic_video_split"],
                self.components["enable_automatic_content_captioning"],
                self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
            ],
            outputs=[
                self.app.tabs_component,
                self.app.tabs["split_tab"].components["video_list"],
                self.app.tabs["split_tab"].components["detect_status"]
            ]
        )
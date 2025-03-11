"""
YouTube tab for Video Model Studio UI.
Handles downloading videos from YouTube URLs.
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..base_tab import BaseTab

logger = logging.getLogger(__name__)

class YouTubeTab(BaseTab):
    """YouTube tab for downloading videos from YouTube"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "youtube_tab"
        self.title = "Download from YouTube"
    
    def create(self, parent=None) -> gr.Tab:
        """Create the YouTube tab UI components"""
        with gr.Tab(self.title, id=self.id) as tab:
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Import a YouTube video")
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("You can use a YouTube video as reference, by pasting its URL here:")
                        with gr.Row():
                            gr.Markdown("Please be aware of the [know limitations](https://stackoverflow.com/questions/78160027/how-to-solve-http-error-400-bad-request-in-pytube) and [issues](https://stackoverflow.com/questions/79226520/pytube-throws-http-error-403-forbidden-since-a-few-days)")

                    with gr.Column():
                        self.components["youtube_url"] = gr.Textbox(
                            label="Import YouTube Video",
                            placeholder="https://www.youtube.com/watch?v=..."
                        )
                
                with gr.Row():
                    self.components["youtube_download_btn"] = gr.Button("Download YouTube Video", variant="primary")
            
            return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # YouTube download event
        self.components["youtube_download_btn"].click(
            fn=self.app.importing.download_youtube_video,
            inputs=[self.components["youtube_url"]],
            outputs=[self.components["import_status"]]  # This comes from parent tab
        ).success(
            fn=self.app.tabs["import_tab"].on_import_success,
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
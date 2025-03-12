"""
YouTube tab for Video Model Studio UI.
Handles downloading videos from YouTube URLs.
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from vms.utils import BaseTab

logger = logging.getLogger(__name__)

class YouTubeTab(BaseTab):
    """YouTube tab for downloading videos from YouTube"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "youtube_tab"
        self.title = "Download from YouTube"
        # Initialize components that will be shared from parent
        if "components" not in self.__dict__:
            self.components = {}
        self.components["import_status"] = None
        self.components["enable_automatic_video_split"] = None
        self.components["enable_automatic_content_captioning"] = None
    
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
                            gr.Markdown("Please be aware of the [known limitations](https://stackoverflow.com/questions/78160027/how-to-solve-http-error-400-bad-request-in-pytube) and [issues](https://stackoverflow.com/questions/79226520/pytube-throws-http-error-403-forbidden-since-a-few-days)")

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
        # Check if required shared components exist before connecting events
        if not self.components.get("import_status"):
            logger.warning("import_status component is not set in YouTubeTab")
            return
        
        if not self.components.get("enable_automatic_video_split"):
            logger.warning("enable_automatic_video_split component is not set in YouTubeTab")
            return
            
        if not self.components.get("enable_automatic_content_captioning"):
            logger.warning("enable_automatic_content_captioning component is not set in YouTubeTab")
            return
            
        # Only try to access custom_prompt_prefix if the caption_tab exists
        custom_prompt_prefix = None
        try:
            if hasattr(self.app.tabs, "caption_tab") and "custom_prompt_prefix" in self.app.tabs["caption_tab"].components:
                custom_prompt_prefix = self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
        except (AttributeError, KeyError):
            logger.warning("Could not access custom_prompt_prefix component")
            
        # Check if we have access to project_tabs_component
        if hasattr(self.app, "project_tabs_component"):
            tabs_component = self.app.project_tabs_component
        else:
            logger.warning("project_tabs_component not found in app, using None for tab switching")
            tabs_component = None
            
        # YouTube download event
        download_event = self.components["youtube_download_btn"].click(
            fn=self.download_youtube_with_splitting,
            inputs=[self.components["youtube_url"], self.components["enable_automatic_video_split"]],
            outputs=[self.components["import_status"]]
        )
        
        # Add success handler if all components exist
        if hasattr(self.app.tabs, "import_tab") and custom_prompt_prefix is not None:
            try:
                # Add the success handler
                download_event.success(
                    fn=self.app.tabs["import_tab"].on_import_success,
                    inputs=[
                        self.components["enable_automatic_video_split"],
                        self.components["enable_automatic_content_captioning"],
                        custom_prompt_prefix
                    ],
                    outputs=[
                        tabs_component,
                        self.components["import_status"]
                    ]
                )
            except (AttributeError, KeyError) as e:
                logger.error(f"Error connecting success handler in YouTubeTab: {str(e)}")
                # Continue without the success handler
                
    def download_youtube_with_splitting(self, url, enable_splitting):
        """Download YouTube video with splitting option"""
        return self.app.importing.download_youtube_video(url, enable_splitting, gr.Progress())
"""
Split tab for Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional

from .base_tab import BaseTab

logger = logging.getLogger(__name__)

class SplitTab(BaseTab):
    """Split tab for scene detection and video splitting"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "split_tab"
        self.title = "2️⃣  Split"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Split tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                self.components["split_title"] = gr.Markdown("## Splitting of 0 videos (0 bytes)")
            
            with gr.Row():
                with gr.Column():
                    self.components["detect_btn"] = gr.Button("Split videos into single-camera shots", variant="primary")
                    self.components["detect_status"] = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    self.components["video_list"] = gr.Dataframe(
                        headers=["name", "status"],
                        label="Videos to split",
                        interactive=False,
                        wrap=True
                    )
                    
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Scene detection button event
        self.components["detect_btn"].click(
            fn=self.app.start_scene_detection,
            inputs=[self.app.tabs["import_tab"].components["enable_automatic_video_split"]],
            outputs=[self.components["detect_status"]]
        )
        
    def refresh(self) -> Dict[str, Any]:
        """Refresh the video list with current data"""
        videos = self.app.splitter.list_unprocessed_videos()
        return {
            "video_list": videos
        }
"""
Split tab for Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional

from vms.utils import BaseTab

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
                        label="Videos to split (note: Nvidia A100 cannot split videos encoded in AV1)",
                        interactive=False,
                        wrap=True
                    )
                    
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Scene detection button event
        self.components["detect_btn"].click(
            fn=self.start_scene_detection,
            inputs=[self.app.tabs["import_tab"].components["enable_automatic_video_split"]],
            outputs=[self.components["detect_status"]]
        )
        
    def refresh(self) -> Dict[str, Any]:
        """Refresh the video list with current data"""
        videos = self.list_unprocessed_videos()
        return {
            "video_list": videos
        }
    
    def list_unprocessed_videos(self) -> gr.Dataframe:
        """Update list of unprocessed videos"""
        videos = self.app.splitting.list_unprocessed_videos()
        # videos is already in [[name, status]] format from splitting_service
        return gr.Dataframe(
            headers=["name", "status"],
            value=videos,
            interactive=False
        )

    async def start_scene_detection(self, enable_splitting: bool) -> str:
        """Start background scene detection process
        
        Args:
            enable_splitting: Whether to split videos into scenes
        """
        if self.app.splitting.is_processing():
            return "Scene detection already running"
            
        try:
            await self.app.splitting.start_processing(enable_splitting)
            return "Scene detection completed"
        except Exception as e:
            return f"Error during scene detection: {str(e)}"
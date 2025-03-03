"""
Split tab for Video Model Studio UI with WebDataset support
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_tab import BaseTab
from ..config import STAGING_PATH

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
        videos = self.app.splitter.list_unprocessed_videos()
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
        if self.app.splitter.is_processing():
            return "Scene detection already running"
            
        try:
            await self.app.splitter.start_processing(enable_splitting)
            # Export processed videos to staging for captioning
            self.export_processed_videos()
            return "Scene detection completed"
        except Exception as e:
            return f"Error during scene detection: {str(e)}"
    
    def export_processed_videos(self) -> str:
        """Export processed videos from WebDataset shards to staging directory
        
        Returns:
            Status message
        """
        try:
            # Use WebDataset processing service to export videos 
            videos, images = self.app.splitter.export_to_staging(STAGING_PATH)
            
            if videos > 0 or images > 0:
                media_types = []
                if videos > 0:
                    media_types.append(f"{videos} video{'s' if videos != 1 else ''}")
                if images > 0:
                    media_types.append(f"{images} image{'s' if images != 1 else ''}")
                
                return f"Exported {' and '.join(media_types)} to staging directory"
            else:
                return "No media to export"
        except Exception as e:
            logger.error(f"Error exporting processed videos: {e}")
            return f"Error exporting videos: {str(e)}"
"""
Manage tab for Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional

from .base_tab import BaseTab
from ..config import HF_API_TOKEN

logger = logging.getLogger(__name__)

class ManageTab(BaseTab):
    """Manage tab for storage management and model publication"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "manage_tab"
        self.title = "5️⃣  Manage"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Manage tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Publishing")
                        gr.Markdown("You model can be pushed to Hugging Face (this will use HF_API_TOKEN)")

                        with gr.Row():
                            with gr.Column():
                                self.components["repo_id"] = gr.Textbox(
                                    label="HuggingFace Model Repository",
                                    placeholder="username/model-name",
                                    info="The repository will be created if it doesn't exist"
                                )
                                self.components["make_public"] = gr.Checkbox(
                                    label="Check this to make your model public (ie. visible and downloadable by anyone)",
                                    info="You model is private by default"
                                )
                                self.components["push_model_btn"] = gr.Button(
                                    "Push my model"
                                )

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("## Storage management")
                                with gr.Row():
                                    self.components["download_dataset_btn"] = gr.DownloadButton(
                                        "Download dataset",
                                        variant="secondary",
                                        size="lg"
                                    )
                                    self.components["download_model_btn"] = gr.DownloadButton(
                                        "Download model",
                                        variant="secondary",
                                        size="lg"
                                    )

                        with gr.Row():
                            self.components["global_stop_btn"] = gr.Button(
                                "Stop everything and delete my data",
                                variant="stop"
                            )
                            self.components["global_status"] = gr.Textbox(
                                label="Global Status",
                                interactive=False,
                                visible=False
                            )
        
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Repository ID validation
        self.components["repo_id"].change(
            fn=self.app.validate_repo,
            inputs=[self.components["repo_id"]],
            outputs=[self.components["repo_id"]]
        )
        
        # Download buttons
        self.components["download_dataset_btn"].click(
            fn=self.app.trainer.create_training_dataset_zip,
            outputs=[self.components["download_dataset_btn"]]
        )

        self.components["download_model_btn"].click(
            fn=self.app.trainer.get_model_output_safetensors,
            outputs=[self.components["download_model_btn"]]
        )
        
        # Global stop button
        self.components["global_stop_btn"].click(
            fn=self.app.handle_global_stop,
            outputs=[
                self.components["global_status"],
                self.app.tabs["split_tab"].components["video_list"],
                self.app.tabs["caption_tab"].components["training_dataset"],
                self.app.tabs["train_tab"].components["status_box"],
                self.app.tabs["train_tab"].components["log_box"],
                self.app.tabs["split_tab"].components["detect_status"],
                self.app.tabs["import_tab"].components["import_status"],
                self.app.tabs["caption_tab"].components["preview_status"]
            ]
        )
        
        # Push model button 
        # To implement model pushing functionality
        self.components["push_model_btn"].click(
            fn=lambda repo_id: self.app.upload_to_hub(repo_id),
            inputs=[self.components["repo_id"]],
            outputs=[self.components["global_status"]]
        )
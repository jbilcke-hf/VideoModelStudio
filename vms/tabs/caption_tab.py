"""
Caption tab for Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_tab import BaseTab
from ..config import DEFAULT_CAPTIONING_BOT_INSTRUCTIONS, DEFAULT_PROMPT_PREFIX

logger = logging.getLogger(__name__)

class CaptionTab(BaseTab):
    """Caption tab for managing asset captions"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "caption_tab"
        self.title = "3️⃣  Caption"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Caption tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                self.components["caption_title"] = gr.Markdown("## Captioning of 0 files (0 bytes)")
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.components["custom_prompt_prefix"] = gr.Textbox(
                            scale=3,
                            label='Prefix to add to ALL captions (eg. "In the style of TOK, ")',
                            placeholder="In the style of TOK, ",
                            lines=2,
                            value=DEFAULT_PROMPT_PREFIX
                        )
                        self.components["captioning_bot_instructions"] = gr.Textbox(
                            scale=6,
                            label="System instructions for the automatic captioning model",
                            placeholder="Please generate a full description of...",
                            lines=5,
                            value=DEFAULT_CAPTIONING_BOT_INSTRUCTIONS
                        )
                    with gr.Row():
                        self.components["run_autocaption_btn"] = gr.Button(
                            "Automatically fill missing captions",
                            variant="primary"
                        )
                        self.components["copy_files_to_training_dir_btn"] = gr.Button(
                            "Copy assets to training directory",
                            variant="primary"
                        )
                        self.components["stop_autocaption_btn"] = gr.Button(
                            "Stop Captioning",
                            variant="stop",
                            interactive=False
                        )

            with gr.Row():
                with gr.Column():
                    self.components["training_dataset"] = gr.Dataframe(
                        headers=["name", "status"],
                        interactive=False,
                        wrap=True,
                        value=self.app.list_training_files_to_caption(),
                        row_count=10
                    )

                with gr.Column():
                    self.components["preview_video"] = gr.Video(
                        label="Video Preview",
                        interactive=False,
                        visible=False
                    )
                    self.components["preview_image"] = gr.Image(
                        label="Image Preview",
                        interactive=False,
                        visible=False
                    )
                    self.components["preview_caption"] = gr.Textbox(
                        label="Caption",
                        lines=6,
                        interactive=True
                    )
                    self.components["save_caption_btn"] = gr.Button("Save Caption")
                    self.components["preview_status"] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=True
                    )
                    self.components["original_file_path"] = gr.State(value=None)
            
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Run auto-captioning button
        self.components["run_autocaption_btn"].click(
            fn=self.app.show_refreshing_status,
            outputs=[self.components["training_dataset"]]
        ).then(
            fn=lambda: self.app.update_captioning_buttons_start(),
            outputs=[
                self.components["run_autocaption_btn"],
                self.components["stop_autocaption_btn"],
                self.components["copy_files_to_training_dir_btn"]
            ]
        ).then(
            fn=self.app.start_caption_generation,
            inputs=[
                self.components["captioning_bot_instructions"],
                self.components["custom_prompt_prefix"]
            ],
            outputs=[self.components["training_dataset"]],
        ).then(
            fn=lambda: self.app.update_captioning_buttons_end(),
            outputs=[
                self.components["run_autocaption_btn"],
                self.components["stop_autocaption_btn"],
                self.components["copy_files_to_training_dir_btn"]
            ]
        )
        
        # Copy files to training dir button
        self.components["copy_files_to_training_dir_btn"].click(
            fn=self.app.copy_files_to_training_dir,
            inputs=[self.components["custom_prompt_prefix"]]
        )
        
        # Stop captioning button
        self.components["stop_autocaption_btn"].click(
            fn=self.app.stop_captioning,
            outputs=[
                self.components["training_dataset"],
                self.components["run_autocaption_btn"],
                self.components["stop_autocaption_btn"],
                self.components["copy_files_to_training_dir_btn"]
            ]
        )
        
        # Dataset selection for preview
        self.components["training_dataset"].select(
            fn=self.app.handle_training_dataset_select,
            outputs=[
                self.components["preview_image"],
                self.components["preview_video"],
                self.components["preview_caption"],
                self.components["original_file_path"],
                self.components["preview_status"]
            ]
        )
        
        # Save caption button
        self.components["save_caption_btn"].click(
            fn=self.app.save_caption_changes,
            inputs=[
                self.components["preview_caption"],
                self.components["preview_image"],
                self.components["preview_video"],
                self.components["original_file_path"],
                self.components["custom_prompt_prefix"]
            ],
            outputs=[self.components["preview_status"]]
        ).success(
            fn=self.app.list_training_files_to_caption,
            outputs=[self.components["training_dataset"]]
        )
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh the dataset list with current data"""
        training_dataset = self.app.list_training_files_to_caption()
        return {
            "training_dataset": training_dataset
        }
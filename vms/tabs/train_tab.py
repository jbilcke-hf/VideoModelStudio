"""
Train tab for Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional

from .base_tab import BaseTab
from ..config import TRAINING_PRESETS, MODEL_TYPES, ASK_USER_TO_DUPLICATE_SPACE
from ..utils import TrainingLogParser

logger = logging.getLogger(__name__)

class TrainTab(BaseTab):
    """Train tab for model training"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "train_tab"
        self.title = "4️⃣  Train"
    
    def handle_training_start(self, preset, model_type, *args):
        """Handle training start with proper log parser reset"""
        # Safely reset log parser if it exists
        if hasattr(self.app, 'log_parser') and self.app.log_parser is not None:
            self.app.log_parser.reset()
        else:
            logger.warning("Log parser not initialized, creating a new one")

            self.app.log_parser = TrainingLogParser()
        
        # Start training
        return self.app.trainer.start_training(
            MODEL_TYPES[model_type],
            *args,
            preset_name=preset
        )
        
    def create(self, parent=None) -> gr.TabItem:
        """Create the Train tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.components["train_title"] = gr.Markdown("## 0 files available for training (0 bytes)")

                    with gr.Row():
                        with gr.Column():
                            self.components["training_preset"] = gr.Dropdown(
                                choices=list(TRAINING_PRESETS.keys()),
                                label="Training Preset",
                                value=list(TRAINING_PRESETS.keys())[0]
                            )
                        self.components["preset_info"] = gr.Markdown()

                    with gr.Row():
                        with gr.Column():
                            self.components["model_type"] = gr.Dropdown(
                                choices=list(MODEL_TYPES.keys()),
                                label="Model Type",
                                value=list(MODEL_TYPES.keys())[0]
                            )
                        self.components["model_info"] = gr.Markdown(
                            value=self.app.get_model_info(list(MODEL_TYPES.keys())[0])
                        )

                    with gr.Row():
                        self.components["lora_rank"] = gr.Dropdown(
                            label="LoRA Rank",
                            choices=["16", "32", "64", "128", "256", "512", "1024"],
                            value="128",
                            type="value"
                        )
                        self.components["lora_alpha"] = gr.Dropdown(
                            label="LoRA Alpha",
                            choices=["16", "32", "64", "128", "256", "512", "1024"],
                            value="128",
                            type="value"
                        )
                    with gr.Row():
                        self.components["num_epochs"] = gr.Number(
                            label="Number of Epochs",
                            value=70,
                            minimum=1,
                            precision=0
                        )
                        self.components["batch_size"] = gr.Number(
                            label="Batch Size",
                            value=1,
                            minimum=1,
                            precision=0
                        )
                    with gr.Row():
                        self.components["learning_rate"] = gr.Number(
                            label="Learning Rate",
                            value=2e-5,
                            minimum=1e-7
                        )
                        self.components["save_iterations"] = gr.Number(
                            label="Save checkpoint every N iterations",
                            value=500,
                            minimum=50,
                            precision=0,
                            info="Model will be saved periodically after these many steps"
                        )
                
                with gr.Column():
                    with gr.Row():
                        self.components["start_btn"] = gr.Button(
                            "Start Training",
                            variant="primary",
                            interactive=not ASK_USER_TO_DUPLICATE_SPACE
                        )
                        self.components["pause_resume_btn"] = gr.Button(
                            "Resume Training",
                            variant="secondary",
                            interactive=False
                        )
                        self.components["stop_btn"] = gr.Button(
                            "Stop Training",
                            variant="stop",
                            interactive=False
                        )

                    with gr.Row():
                        with gr.Column():
                            self.components["status_box"] = gr.Textbox(
                                label="Training Status",
                                interactive=False,
                                lines=4
                            )
                            with gr.Accordion("See training logs"):
                                self.components["log_box"] = gr.TextArea(
                                    label="Finetrainers output (see HF Space logs for more details)",
                                    interactive=False,
                                    lines=40,
                                    max_lines=200,
                                    autoscroll=True
                                )
                    
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Model type change event
        def update_model_info(model):
            params = self.app.get_default_params(MODEL_TYPES[model])
            info = self.app.get_model_info(MODEL_TYPES[model])
            return {
                self.components["model_info"]: info,
                self.components["num_epochs"]: params["num_epochs"],
                self.components["batch_size"]: params["batch_size"],
                self.components["learning_rate"]: params["learning_rate"],
                self.components["save_iterations"]: params["save_iterations"]
            }
            
        self.components["model_type"].change(
            fn=lambda v: self.app.update_ui_state(model_type=v),
            inputs=[self.components["model_type"]],
            outputs=[]
        ).then(
            fn=update_model_info,
            inputs=[self.components["model_type"]],
            outputs=[
                self.components["model_info"],
                self.components["num_epochs"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"]
            ]
        )
        
        # Training parameters change events
        self.components["lora_rank"].change(
            fn=lambda v: self.app.update_ui_state(lora_rank=v),
            inputs=[self.components["lora_rank"]],
            outputs=[]
        )

        self.components["lora_alpha"].change(
            fn=lambda v: self.app.update_ui_state(lora_alpha=v),
            inputs=[self.components["lora_alpha"]],
            outputs=[]
        )

        self.components["num_epochs"].change(
            fn=lambda v: self.app.update_ui_state(num_epochs=v),
            inputs=[self.components["num_epochs"]],
            outputs=[]
        )

        self.components["batch_size"].change(
            fn=lambda v: self.app.update_ui_state(batch_size=v),
            inputs=[self.components["batch_size"]],
            outputs=[]
        )

        self.components["learning_rate"].change(
            fn=lambda v: self.app.update_ui_state(learning_rate=v),
            inputs=[self.components["learning_rate"]],
            outputs=[]
        )

        self.components["save_iterations"].change(
            fn=lambda v: self.app.update_ui_state(save_iterations=v),
            inputs=[self.components["save_iterations"]],
            outputs=[]
        )
        
        # Training preset change event
        self.components["training_preset"].change(
            fn=lambda v: self.app.update_ui_state(training_preset=v),
            inputs=[self.components["training_preset"]],
            outputs=[]
        ).then(
            fn=self.app.update_training_params,
            inputs=[self.components["training_preset"]],
            outputs=[
                self.components["model_type"],
                self.components["lora_rank"],
                self.components["lora_alpha"],
                self.components["num_epochs"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"],
                self.components["preset_info"]
            ]
        )
        
        # Training control events
        self.components["start_btn"].click(
            fn=self.handle_training_start,  # Use safer method instead of lambda
            inputs=[
                self.components["training_preset"],
                self.components["model_type"],
                self.components["lora_rank"],
                self.components["lora_alpha"],
                self.components["num_epochs"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"],
                self.app.tabs["manage_tab"].components["repo_id"]
            ],
            outputs=[
                self.components["status_box"],
                self.components["log_box"]
            ]
        ).success(
            fn=self.app.get_latest_status_message_logs_and_button_labels,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                self.components["pause_resume_btn"]
            ]
        )

        self.components["pause_resume_btn"].click(
            fn=self.app.handle_pause_resume,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                self.components["pause_resume_btn"]
            ]
        )

        self.components["stop_btn"].click(
            fn=self.app.handle_stop,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                self.components["pause_resume_btn"]
            ]
        )
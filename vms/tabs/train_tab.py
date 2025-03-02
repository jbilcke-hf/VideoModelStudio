"""
Train tab for Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .base_tab import BaseTab
from ..config import TRAINING_PRESETS, MODEL_TYPES, ASK_USER_TO_DUPLICATE_SPACE, SMALL_TRAINING_BUCKETS
from ..utils import TrainingLogParser

logger = logging.getLogger(__name__)

class TrainTab(BaseTab):
    """Train tab for model training"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "train_tab"
        self.title = "4️⃣  Train"
    
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
                            value=self.get_model_info(list(MODEL_TYPES.keys())[0])
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
                        # Check for existing checkpoints to determine button text
                        has_checkpoints = len(list(OUTPUT_PATH.glob("checkpoint-*"))) > 0
                        start_text = "Continue Training" if has_checkpoints else "Start Training"
                        
                        self.components["start_btn"] = gr.Button(
                            start_text,
                            variant="primary",
                            interactive=not ASK_USER_TO_DUPLICATE_SPACE
                        )
                        
                        # Just use stop and pause buttons for now to ensure compatibility
                        self.components["stop_btn"] = gr.Button(
                            "Stop at Last Checkpoint",
                            variant="primary",
                            interactive=False
                        )
                        
                        self.components["pause_resume_btn"] = gr.Button(
                            "Resume Training",
                            variant="secondary",
                            interactive=False,
                            visible=False
                        )
                        
                        # Add delete checkpoints button - THIS IS THE KEY FIX
                        self.components["delete_checkpoints_btn"] = gr.Button(
                            "Delete All Checkpoints",
                            variant="stop",
                            interactive=True
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
            params = self.get_default_params(MODEL_TYPES[model])
            info = self.get_model_info(MODEL_TYPES[model])
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
            fn=self.update_training_params,
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
            fn=self.handle_training_start,
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
            fn=self.get_latest_status_message_logs_and_button_labels,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                self.components["pause_resume_btn"]
            ]
        )

        self.components["pause_resume_btn"].click(
            fn=self.handle_pause_resume,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                self.components["pause_resume_btn"]
            ]
        )

        self.components["stop_btn"].click(
            fn=self.handle_stop,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                self.components["pause_resume_btn"]
            ]
        )
        
    def handle_training_start(self, preset, model_type, *args):
        """Handle training start with proper log parser reset"""
        # Safely reset log parser if it exists
        if hasattr(self.app, 'log_parser') and self.app.log_parser is not None:
            self.app.log_parser.reset()
        else:
            logger.warning("Log parser not initialized, creating a new one")
            from ..utils import TrainingLogParser
            self.app.log_parser = TrainingLogParser()
        
        # Start training
        return self.app.trainer.start_training(
            MODEL_TYPES[model_type],
            *args,
            preset_name=preset
        )
    
    def get_model_info(self, model_type: str) -> str:
        """Get information about the selected model type"""
        if model_type == "hunyuan_video":
            return """### HunyuanVideo (LoRA)
    - Required VRAM: ~48GB minimum
    - Recommended batch size: 1-2
    - Typical training time: 2-4 hours
    - Default resolution: 49x512x768
    - Default LoRA rank: 128 (~600 MB)"""
                
        elif model_type == "ltx_video":
            return """### LTX-Video (LoRA)
    - Required VRAM: ~18GB minimum 
    - Recommended batch size: 1-4
    - Typical training time: 1-3 hours
    - Default resolution: 49x512x768
    - Default LoRA rank: 128"""
                
        return ""

    def get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default training parameters for model type"""
        if model_type == "hunyuan_video":
            return {
                "num_epochs": 70,
                "batch_size": 1,
                "learning_rate": 2e-5,
                "save_iterations": 500,
                "video_resolution_buckets": SMALL_TRAINING_BUCKETS,
                "video_reshape_mode": "center",
                "caption_dropout_p": 0.05,
                "gradient_accumulation_steps": 1,
                "rank": 128,
                "lora_alpha": 128
            }
        else:  # ltx_video
            return {
                "num_epochs": 70,
                "batch_size": 1,
                "learning_rate": 3e-5,
                "save_iterations": 500,
                "video_resolution_buckets": SMALL_TRAINING_BUCKETS,
                "video_reshape_mode": "center",
                "caption_dropout_p": 0.05,
                "gradient_accumulation_steps": 4,
                "rank": 128,
                "lora_alpha": 128
            }
            
    def update_training_params(self, preset_name: str) -> Tuple:
        """Update UI components based on selected preset while preserving custom settings"""
        preset = TRAINING_PRESETS[preset_name]
        
        # Load current UI state to check if user has customized values
        current_state = self.app.load_ui_values()
        
        # Find the display name that maps to our model type
        model_display_name = next(
            key for key, value in MODEL_TYPES.items() 
            if value == preset["model_type"]
        )
            
        # Get preset description for display
        description = preset.get("description", "")
        
        # Get max values from buckets
        buckets = preset["training_buckets"]
        max_frames = max(frames for frames, _, _ in buckets)
        max_height = max(height for _, height, _ in buckets)
        max_width = max(width for _, _, width in buckets)
        bucket_info = f"\nMaximum video size: {max_frames} frames at {max_width}x{max_height} resolution"
        
        info_text = f"{description}{bucket_info}"
        
        # Return values in the same order as the output components
        # Use preset defaults but preserve user-modified values if they exist
        lora_rank_val = current_state.get("lora_rank") if current_state.get("lora_rank") != preset.get("lora_rank", "128") else preset["lora_rank"]
        lora_alpha_val = current_state.get("lora_alpha") if current_state.get("lora_alpha") != preset.get("lora_alpha", "128") else preset["lora_alpha"]
        num_epochs_val = current_state.get("num_epochs") if current_state.get("num_epochs") != preset.get("num_epochs", 70) else preset["num_epochs"]
        batch_size_val = current_state.get("batch_size") if current_state.get("batch_size") != preset.get("batch_size", 1) else preset["batch_size"]
        learning_rate_val = current_state.get("learning_rate") if current_state.get("learning_rate") != preset.get("learning_rate", 3e-5) else preset["learning_rate"]
        save_iterations_val = current_state.get("save_iterations") if current_state.get("save_iterations") != preset.get("save_iterations", 500) else preset["save_iterations"]
        
        return (
            model_display_name,
            lora_rank_val,
            lora_alpha_val,
            num_epochs_val,
            batch_size_val,
            learning_rate_val,
            save_iterations_val,
            info_text
        )
    
    def update_training_ui(self, training_state: Dict[str, Any]):
        """Update UI components based on training state"""
        updates = {}
        
        # Update status box with high-level information
        status_text = []
        if training_state["status"] != "idle":
            status_text.extend([
                f"Status: {training_state['status']}",
                f"Progress: {training_state['progress']}",
                f"Step: {training_state['current_step']}/{training_state['total_steps']}",
                    
                # Epoch information
                # there is an issue with how epoch is reported because we display:
                # Progress: 96.9%, Step: 872/900, Epoch: 12/50
                # we should probably just show the steps
                #f"Epoch: {training_state['current_epoch']}/{training_state['total_epochs']}",
                
                f"Time elapsed: {training_state['elapsed']}",
                f"Estimated remaining: {training_state['remaining']}",
                "",
                f"Current loss: {training_state['step_loss']}",
                f"Learning rate: {training_state['learning_rate']}",
                f"Gradient norm: {training_state['grad_norm']}",
                f"Memory usage: {training_state['memory']}"
            ])
            
            if training_state["error_message"]:
                status_text.append(f"\nError: {training_state['error_message']}")
                
        updates["status_box"] = "\n".join(status_text)
        
        # Update button states
        updates["start_btn"] = gr.Button(
            "Start training",
            interactive=(training_state["status"] in ["idle", "completed", "error", "stopped"]),
            variant="primary" if training_state["status"] == "idle" else "secondary"
        )
        
        updates["stop_btn"] = gr.Button(
            "Stop training",
            interactive=(training_state["status"] in ["training", "initializing"]),
            variant="stop"
        )
        
        return updates
        
    def handle_pause_resume(self):
        status, _, _ = self.get_latest_status_message_and_logs()

        if status == "paused":
            self.app.trainer.resume_training()
        else:
            self.app.trainer.pause_training()

        return self.get_latest_status_message_logs_and_button_labels()

    def handle_stop(self):
        self.app.trainer.stop_training()
        return self.get_latest_status_message_logs_and_button_labels()
    
    def get_latest_status_message_and_logs(self) -> Tuple[str, str, str]:
        """Get latest status message, log content, and status code in a safer way"""
        state = self.app.trainer.get_status()
        logs = self.app.trainer.get_logs()

        # Ensure log parser is initialized
        if not hasattr(self.app, 'log_parser') or self.app.log_parser is None:
            from ..utils import TrainingLogParser
            self.app.log_parser = TrainingLogParser()
            logger.info("Initialized missing log parser")

        # Parse new log lines
        if logs:
            last_state = None
            for line in logs.splitlines():
                try:
                    state_update = self.app.log_parser.parse_line(line)
                    if state_update:
                        last_state = state_update
                except Exception as e:
                    logger.error(f"Error parsing log line: {str(e)}")
                    continue
            
            if last_state:
                ui_updates = self.update_training_ui(last_state)
                state["message"] = ui_updates.get("status_box", state["message"])
        
        # Parse status for training state
        if "completed" in state["message"].lower():
            state["status"] = "completed"

        return (state["status"], state["message"], logs)

    def get_latest_status_message_logs_and_button_labels(self) -> Tuple:
        """Get latest status message, logs and button states"""
        status, message, logs = self.get_latest_status_message_and_logs()
        
        # Add checkpoints detection
        has_checkpoints = len(list(OUTPUT_PATH.glob("checkpoint-*"))) > 0
        
        button_updates = self.update_training_buttons(status, has_checkpoints).values()
        
        # Return in order expected by timer
        return (message, logs, *button_updates)
    
    def update_training_buttons(self, status: str, has_checkpoints: bool = None) -> Dict:
        """Update training control buttons based on state"""
        if has_checkpoints is None:
            has_checkpoints = len(list(OUTPUT_PATH.glob("checkpoint-*"))) > 0
            
        is_training = status in ["training", "initializing"]
        is_completed = status in ["completed", "error", "stopped"]
        
        start_text = "Continue Training" if has_checkpoints else "Start Training"
        
        # Only include buttons that we know exist in components
        result = {
            "start_btn": gr.Button(
                value=start_text,
                interactive=not is_training,
                variant="primary" if not is_training else "secondary",
            ),
            "stop_btn": gr.Button(
                value="Stop at Last Checkpoint",
                interactive=is_training,
                variant="primary" if is_training else "secondary",
            )
        }
        
        # Add delete_checkpoints_btn only if it exists in components
        if "delete_checkpoints_btn" in self.components:
            result["delete_checkpoints_btn"] = gr.Button(
                value="Delete All Checkpoints",
                interactive=has_checkpoints and not is_training,
                variant="stop",
            )
        else:
            # Add pause_resume_btn as fallback
            result["pause_resume_btn"] = gr.Button(
                value="Resume Training" if status == "paused" else "Pause Training",
                interactive=(is_training or status == "paused") and not is_completed,
                variant="secondary",
                visible=False
            )
        
        return result
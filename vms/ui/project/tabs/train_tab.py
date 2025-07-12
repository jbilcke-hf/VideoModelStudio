"""
Train tab for Video Model Studio UI with improved task progress display
"""

import gradio as gr
import logging
import os
import json
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from vms.utils import BaseTab
from vms.config import (
    ASK_USER_TO_DUPLICATE_SPACE,
    SD_TRAINING_BUCKETS, MD_TRAINING_BUCKETS,
    RESOLUTION_OPTIONS,
    TRAINING_TYPES, MODEL_TYPES, MODEL_VERSIONS,
    DEFAULT_NB_TRAINING_STEPS, DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
    DEFAULT_BATCH_SIZE, DEFAULT_CAPTION_DROPOUT_P,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK_STR, DEFAULT_LORA_ALPHA_STR,
    DEFAULT_SEED,
    DEFAULT_NUM_GPUS,
    DEFAULT_MAX_GPUS,
    DEFAULT_PRECOMPUTATION_ITEMS,
    DEFAULT_NB_TRAINING_STEPS,
    DEFAULT_NB_LR_WARMUP_STEPS,
    DEFAULT_AUTO_RESUME,
    DEFAULT_CONTROL_TYPE, DEFAULT_TRAIN_QK_NORM,
    DEFAULT_FRAME_CONDITIONING_TYPE, DEFAULT_FRAME_CONDITIONING_INDEX,
    DEFAULT_FRAME_CONDITIONING_CONCATENATE_MASK,
    HUNYUAN_VIDEO_DEFAULTS, LTX_VIDEO_DEFAULTS, WAN_DEFAULTS
)

logger = logging.getLogger(__name__)

class TrainTab(BaseTab):
    """Train tab for model training"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "train_tab"
        self.title = "3️⃣ Train"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Train tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.components["train_title"] = gr.Markdown("## 0 files in the training dataset")

                    with gr.Row():
                        with gr.Column():
                            # Get the default model type from the first preset
                            default_model_type = list(MODEL_TYPES.keys())[0]

                            self.components["model_type"] = gr.Dropdown(
                                choices=list(MODEL_TYPES.keys()),
                                label="Model Type",
                                value=default_model_type,
                                interactive=True
                            )

                            # Get model versions for the default model type
                            default_model_versions = self.get_model_version_choices(default_model_type)
                            default_model_version = self.get_default_model_version(default_model_type)

                            # Ensure default_model_versions is not empty
                            if not default_model_versions:
                                # If no versions found for default model, use a fallback
                                internal_type = MODEL_TYPES.get(default_model_type)
                                if internal_type in MODEL_VERSIONS:
                                    default_model_versions = list(MODEL_VERSIONS[internal_type].keys())
                                else:
                                    # Last resort - collect all available versions from all models
                                    default_model_versions = []
                                    for model_versions in MODEL_VERSIONS.values():
                                        default_model_versions.extend(list(model_versions.keys()))
                                    
                                # If still empty, provide a placeholder
                                if not default_model_versions:
                                    default_model_versions = ["-- No versions available --"]
                                    
                                # Set default version to first in list if available
                                if default_model_versions:
                                    default_model_version = default_model_versions[0]
                                else:
                                    default_model_version = ""

                            self.components["model_version"] = gr.Dropdown(
                                choices=default_model_versions,
                                label="Model Version",
                                value=default_model_version,
                                interactive=True,
                                allow_custom_value=True  # Add this to avoid errors with custom values
                            )
                            
                            self.components["training_type"] = gr.Dropdown(
                                choices=list(TRAINING_TYPES.keys()),
                                label="Training Type",
                                value=list(TRAINING_TYPES.keys())[0]
                            )

                    with gr.Row():
                        self.components["model_info"] = gr.Markdown(
                            value=self.get_model_info(list(MODEL_TYPES.keys())[0], list(TRAINING_TYPES.keys())[0])
                        )
                    
                    with gr.Row():
                        with gr.Column():
                            self.components["resolution"] = gr.Dropdown(
                                choices=list(RESOLUTION_OPTIONS.keys()),
                                label="Resolution",
                                value=list(RESOLUTION_OPTIONS.keys())[0],
                                info="Select the resolution for training videos"
                            )

                    # LoRA specific parameters (will show/hide based on training type)
                    with gr.Row(visible=True) as lora_params_row:
                        self.components["lora_params_row"] = lora_params_row
                        with gr.Column():
                            gr.Markdown("""
                            ## 🔄 LoRA Training Parameters
                            
                            LoRA (Low-Rank Adaptation) trains small adapter matrices instead of the full model, requiring much less memory while still achieving great results.
                            """)
                    
                    # Second row for actual LoRA parameters
                    with gr.Row(visible=True) as lora_settings_row:
                        self.components["lora_settings_row"] = lora_settings_row
                        with gr.Column():
                            self.components["lora_rank"] = gr.Dropdown(
                                label="LoRA Rank",
                                choices=["16", "32", "64", "128", "256", "512", "1024"],
                                value=DEFAULT_LORA_RANK_STR,
                                type="value",
                                info="Controls the size and expressiveness of LoRA adapters. Higher values = better quality but larger file size"
                            )
                            
                            with gr.Accordion("What is LoRA Rank?", open=False):
                                gr.Markdown("""
**LoRA Rank** determines the complexity of the LoRA adapters:

- **Lower rank (16-32)**: Smaller file size, faster training, but less expressive
- **Medium rank (64-128)**: Good balance between quality and file size
- **Higher rank (256-1024)**: More expressive adapters, better quality but larger file size

Think of rank as the "capacity" of your adapter. Higher ranks can learn more complex modifications to the base model but require more VRAM during training and result in larger files.

**Quick guide:**
- For Wan models: Use 32-64 (Wan models work well with lower ranks)
- For LTX-Video: Use 128-256
- For Hunyuan Video: Use 128
                                """)
                        
                        with gr.Column():
                            self.components["lora_alpha"] = gr.Dropdown(
                                label="LoRA Alpha",
                                choices=["16", "32", "64", "128", "256", "512", "1024"],
                                value=DEFAULT_LORA_ALPHA_STR,
                                type="value",
                                info="Controls the effective learning rate scaling of LoRA adapters. Usually set to same value as rank"
                            )
                            with gr.Accordion("What is LoRA Alpha?", open=False):
                                gr.Markdown("""
**LoRA Alpha** controls the effective scale of the LoRA updates:

- The actual scaling factor is calculated as `alpha ÷ rank`
- Usually set to match the rank value (alpha = rank)
- Higher alpha = stronger effect from the adapters
- Lower alpha = more subtle adapter influence

**Best practice:**
- For most cases, set alpha equal to rank
- For more aggressive training, set alpha higher than rank
- For more conservative training, set alpha lower than rank
                                """)
                                
                        
                    # Control specific parameters (will show/hide based on training type)
                    with gr.Row(visible=False) as control_params_row:
                        self.components["control_params_row"] = control_params_row
                        with gr.Column():
                            gr.Markdown("""
## 🖼️ Control Training Settings

Control training enables **image-to-video generation** by teaching the model how to use an image as a guide for video creation. 
This is ideal for turning still images into dynamic videos while preserving composition, style, and content.
                            """)
                    
                    # Second row for control parameters
                    with gr.Row(visible=False) as control_settings_row:
                        self.components["control_settings_row"] = control_settings_row
                        with gr.Column():
                            self.components["control_type"] = gr.Dropdown(
                                label="Control Type",
                                choices=["canny", "custom"],
                                value=DEFAULT_CONTROL_TYPE,
                                info="Type of control conditioning. 'canny' uses edge detection preprocessing, 'custom' allows direct image conditioning."
                            )
                            
                            with gr.Accordion("What is Control Conditioning?", open=False):
                                gr.Markdown("""
**Control Conditioning** allows the model to be guided by an input image, adapting the video generation based on the image content. This is used for image-to-video generation where you want to turn an image into a moving video while maintaining its style, composition or content.

- **canny**: Uses edge detection to extract outlines from images for structure-preserving video generation
- **custom**: Direct image conditioning without preprocessing, preserving more image details
                                """)
                        
                        with gr.Column():
                            self.components["train_qk_norm"] = gr.Checkbox(
                                label="Train QK Normalization Layers",
                                value=DEFAULT_TRAIN_QK_NORM,
                                info="Enable to train query-key normalization layers for better control signal integration"
                            )
                            
                            with gr.Accordion("What is QK Normalization?", open=False):
                                gr.Markdown("""
**QK Normalization** refers to normalizing the query and key values in the attention mechanism of transformers.

- When enabled, allows the model to better integrate control signals with content generation
- Improves training stability for control models
- Generally recommended for control training, especially with image conditioning
                                """)
                    
                    with gr.Row(visible=False) as frame_conditioning_row:
                        self.components["frame_conditioning_row"] = frame_conditioning_row
                        with gr.Column():
                            self.components["frame_conditioning_type"] = gr.Dropdown(
                                label="Frame Conditioning Type",
                                choices=["index", "prefix", "random", "first_and_last", "full"],
                                value=DEFAULT_FRAME_CONDITIONING_TYPE,
                                info="Determines which frames receive conditioning during training"
                            )
                            
                            with gr.Accordion("Frame Conditioning Type Explanation", open=False):
                                gr.Markdown("""
**Frame Conditioning Types** determine which frames in the video receive image conditioning:

- **index**: Only applies conditioning to a single frame at the specified index
- **prefix**: Applies conditioning to all frames before a certain point
- **random**: Randomly selects frames to receive conditioning during training
- **first_and_last**: Only applies conditioning to the first and last frames
- **full**: Applies conditioning to all frames in the video

For image-to-video tasks, 'index' (usually with index 0) is most common as it conditions only the first frame.
                                """)
                        
                        with gr.Column():
                            self.components["frame_conditioning_index"] = gr.Number(
                                label="Frame Conditioning Index",
                                value=DEFAULT_FRAME_CONDITIONING_INDEX,
                                precision=0,
                                info="Specifies which frame receives conditioning when using 'index' type (0 = first frame)"
                            )
                    
                    with gr.Row(visible=False) as control_options_row:
                        self.components["control_options_row"] = control_options_row
                        with gr.Column():
                            self.components["frame_conditioning_concatenate_mask"] = gr.Checkbox(
                                label="Concatenate Frame Mask",
                                value=DEFAULT_FRAME_CONDITIONING_CONCATENATE_MASK,
                                info="Enable to add frame mask information to the conditioning channels"
                            )
                            
                            with gr.Accordion("What is Frame Mask Concatenation?", open=False):
                                gr.Markdown("""
**Frame Mask Concatenation** adds an additional channel to the control signal that indicates which frames are being conditioned:
                                
- Creates a binary mask (0/1) indicating which frames receive conditioning
- Helps the model distinguish between conditioned and unconditioned frames
- Particularly useful for 'index' conditioning where only select frames are conditioned
- Generally improves temporal consistency between conditioned and unconditioned frames
                                """)
                                
                        with gr.Column():
                            # Empty column for layout balance
                            pass
                    
                    with gr.Row():
                        self.components["train_steps"] = gr.Number(
                            label="Number of Training Steps",
                            value=DEFAULT_NB_TRAINING_STEPS,
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
                            value=DEFAULT_LEARNING_RATE,
                            minimum=1e-8
                        )
                        self.components["save_iterations"] = gr.Number(
                            label="Save checkpoint every N iterations",
                            value=DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
                            minimum=1,
                            precision=0,
                            info="Model will be saved periodically after these many steps"
                        )
                    with gr.Row():
                        self.components["num_gpus"] = gr.Slider(
                            label="Number of GPUs to use",
                            value=DEFAULT_NUM_GPUS,
                            minimum=1,
                            maximum=DEFAULT_MAX_GPUS,
                            step=1,
                            info="Number of GPUs to use for training"
                        )
                        self.components["precomputation_items"] = gr.Number(
                            label="Precomputation Items",
                            value=DEFAULT_PRECOMPUTATION_ITEMS,
                            minimum=1,
                            precision=0,
                            info="Should be more or less the number of total items (ex: 200 videos), divided by the number of GPUs"
                        )
                    with gr.Row():
                        self.components["lr_warmup_steps"] = gr.Number(
                            label="Learning Rate Warmup Steps",
                            value=DEFAULT_NB_LR_WARMUP_STEPS,
                            minimum=0,
                            precision=0,
                            info="Number of warmup steps (typically 20-40% of total training steps). This helps reducing the impact of early training examples as well as giving time to optimizers to compute accurate statistics."
                        )

                with gr.Row():
                    with gr.Column():

                        with gr.Row():  
                            with gr.Column():
                                # Add description of the training buttons
                                self.components["training_buttons_info"] = gr.Markdown("""
                                ## ⚗️ Train your model on your dataset
                                - **🚀 Start new training**: Begins training from scratch (clears previous checkpoints)
                                - **🛸 Start from latest checkpoint**: Continues training from the most recent checkpoint
                                """)

                                #Finetrainers doesn't support recovery of a training session using a LoRA,
                                #so this feature doesn't work, I've disabled the line/documentation:
                                #- **🔄 Start over using latest LoRA weights**: Start fresh training but use existing LoRA weights as initialization
                      
                                with gr.Row():
                                    # Check for existing checkpoints to determine button text
                                    checkpoints = list(self.app.output_path.glob("finetrainers_step_*"))
                                    has_checkpoints = len(checkpoints) > 0
                                    
                                    # Check for existing LoRA weights
                                    lora_weights_path = self.app.output_path / "lora_weights"
                                    has_lora_weights = False
                                    if lora_weights_path.exists():
                                        lora_dirs = [d for d in lora_weights_path.iterdir() if d.is_dir()]
                                        has_lora_weights = len(lora_dirs) > 0

                                    self.components["start_btn"] = gr.Button(
                                        "🚀 Start new training",
                                        variant="primary",
                                        interactive=not ASK_USER_TO_DUPLICATE_SPACE
                                    )
                                    
                                    # Add new button for continuing from checkpoint
                                    self.components["resume_btn"] = gr.Button(
                                        "🛸 Start from latest checkpoint",
                                        variant="primary", 
                                        interactive=has_checkpoints and not ASK_USER_TO_DUPLICATE_SPACE
                                    )
                                    
                                    # Starting from LoRA weights is DISABLED for now
                                    self.components["start_from_lora_btn"] = gr.Button(
                                        "🔄 Start over using latest LoRA weights",
                                        variant="secondary",
                                        interactive=has_lora_weights and not ASK_USER_TO_DUPLICATE_SPACE,
                                        visible=False,
                                    )
                                    
                                with gr.Row():
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
                                    
                                    # Add delete checkpoints button
                                    self.components["delete_checkpoints_btn"] = gr.Button(
                                        "Delete All Checkpoints",
                                        variant="stop",
                                        interactive=has_checkpoints
                                    )

                                with gr.Row():
                                    self.components["auto_resume"] = gr.Checkbox(
                                        label="Automatically continue training in case of server reboot.",
                                        value=DEFAULT_AUTO_RESUME,
                                        info="When enabled, training will automatically resume from the latest checkpoint after app restart"
                                    )

                        with gr.Row():
                            with gr.Column():
                                self.components["status_box"] = gr.Textbox(
                                    label="Training Status",
                                    interactive=False,
                                    lines=4
                                )
                                
                                # Add new component for current task progress
                                self.components["current_task_box"] = gr.Textbox(
                                    label="Current Task Progress",
                                    interactive=False,
                                    lines=3,
                                    elem_id="current_task_display"
                                )
                                
                                with gr.Accordion("Finetrainers output (or see app logs for more details)", open=False):
                                    self.components["log_box"] = gr.TextArea(
                                        #label="",
                                        interactive=False,
                                        lines=60,
                                        max_lines=600,
                                        autoscroll=True
                                    )
                    
        return tab
    
    def update_model_type_and_version(self, model_type: str, model_version: str):
        """Update both model type and version together to keep them in sync"""
        # Get internal model type
        internal_type = MODEL_TYPES.get(model_type)
        
        # Make sure model_version is valid for this model type
        if internal_type and internal_type in MODEL_VERSIONS:
            valid_versions = list(MODEL_VERSIONS[internal_type].keys())
            if not model_version or model_version not in valid_versions:
                if valid_versions:
                    model_version = valid_versions[0]
        
        # Update UI state with both values to keep them in sync
        self.app.update_ui_state(model_type=model_type, model_version=model_version)
        return None

    def save_model_version(self, model_type: str, model_version: str):
        """Save model version ensuring it's consistent with model type"""
        internal_type = MODEL_TYPES.get(model_type)
        
        # Verify the model_version is compatible with the current model_type
        if internal_type and internal_type in MODEL_VERSIONS:
            valid_versions = MODEL_VERSIONS[internal_type].keys()
            if model_version not in valid_versions:
                # Don't save incompatible version
                return None
                
        # Save the model version along with current model type to ensure consistency
        self.app.update_ui_state(model_type=model_type, model_version=model_version)
        return None

    def handle_new_training_start(
        self, model_type, model_version, training_type, 
        lora_rank, lora_alpha, train_steps, batch_size, learning_rate, 
        save_iterations, repo_id, progress=gr.Progress()
    ):
        """Handle new training start with checkpoint cleanup"""
        # Clear output directory to start fresh

        # Delete all checkpoint directories
        for checkpoint in self.app.output_path.glob("finetrainers_step_*"):
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)
                
        # Also delete session.json which contains previous training info
        session_file = self.app.output_path / "session.json"
        if session_file.exists():
            session_file.unlink()
        
        self.app.training.append_log("Cleared previous checkpoints for new training session")
        
        # Start training normally
        status, logs = self.handle_training_start(
            model_type, model_version, training_type, 
            lora_rank, lora_alpha, train_steps, batch_size, learning_rate, 
            save_iterations, repo_id, progress
        )
        
        # Update download button texts
        manage_tab = self.app.tabs["manage_tab"]
        download_btn_text = gr.update(value=manage_tab.get_download_button_text())
        checkpoint_btn_text = gr.update(value=manage_tab.get_checkpoint_button_text())
        
        return status, logs, download_btn_text, checkpoint_btn_text

    def handle_resume_training(
        self, model_type, model_version, training_type, 
        lora_rank, lora_alpha, train_steps, batch_size, learning_rate, 
        save_iterations, repo_id, progress=gr.Progress()
    ):
        """Handle resuming training from the latest checkpoint"""
        # Find the latest checkpoint
        checkpoints = list(self.app.output_path.glob("finetrainers_step_*"))

        if not checkpoints:
            manage_tab = self.app.tabs["manage_tab"]
            download_btn_text = gr.update(value=manage_tab.get_download_button_text())
            checkpoint_btn_text = gr.update(value=manage_tab.get_checkpoint_button_text())
            return "No checkpoints found to resume from", "Please start a new training session instead", download_btn_text, checkpoint_btn_text
        
        self.app.training.append_log(f"Resuming training from latest checkpoint")
    
        # Start training with the checkpoint
        status, logs = self.handle_training_start(
            model_type, model_version, training_type, 
            lora_rank, lora_alpha, train_steps, batch_size, learning_rate, 
            save_iterations, repo_id, progress, 
            resume_from_checkpoint="latest"
        )
        
        # Update download button texts
        manage_tab = self.app.tabs["manage_tab"]
        download_btn_text = gr.update(value=manage_tab.get_download_button_text())
        checkpoint_btn_text = gr.update(value=manage_tab.get_checkpoint_button_text())
        
        return status, logs, download_btn_text, checkpoint_btn_text

    def handle_start_from_lora_training(
        self, model_type, model_version, training_type, 
        lora_rank, lora_alpha, train_steps, batch_size, learning_rate, 
        save_iterations, repo_id, progress=gr.Progress()
    ):
        """Handle starting training from existing LoRA weights"""
        # Find the latest LoRA weights
        lora_weights_path = self.app.output_path / "lora_weights"
        
        manage_tab = self.app.tabs["manage_tab"]
        download_btn_text = gr.update(value=manage_tab.get_download_button_text())
        checkpoint_btn_text = gr.update(value=manage_tab.get_checkpoint_button_text())
        
        if not lora_weights_path.exists():
            return "No LoRA weights found", "Please train a model first or start a new training session", download_btn_text, checkpoint_btn_text
        
        # Find the latest LoRA checkpoint directory
        lora_dirs = sorted([d for d in lora_weights_path.iterdir() if d.is_dir()], 
                          key=lambda x: int(x.name), reverse=True)
        
        if not lora_dirs:
            return "No LoRA weight directories found", "Please train a model first or start a new training session", download_btn_text, checkpoint_btn_text
        
        latest_lora_dir = lora_dirs[0]
        
        # Verify the LoRA weights file exists
        lora_weights_file = latest_lora_dir / "pytorch_lora_weights.safetensors"
        if not lora_weights_file.exists():
            return f"LoRA weights file not found in {latest_lora_dir}", "Please check your LoRA weights directory", download_btn_text, checkpoint_btn_text
        
        # Clear checkpoints to start fresh (but keep LoRA weights)
        for checkpoint in self.app.output_path.glob("finetrainers_step_*"):
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)
                
        # Delete session.json to start fresh
        session_file = self.app.output_path / "session.json"
        if session_file.exists():
            session_file.unlink()
        
        self.app.training.append_log(f"Starting training from LoRA weights: {latest_lora_dir}")
        
        # Start training with the LoRA weights
        status, logs = self.handle_training_start(
            model_type, model_version, training_type, 
            lora_rank, lora_alpha, train_steps, batch_size, learning_rate, 
            save_iterations, repo_id, progress,
        )
        
        # Update download button texts
        download_btn_text = gr.update(value=manage_tab.get_download_button_text())
        checkpoint_btn_text = gr.update(value=manage_tab.get_checkpoint_button_text())
        
        return status, logs, download_btn_text, checkpoint_btn_text

    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Model type change event - Update model version dropdown choices and default parameters
        self.components["model_type"].change(
            fn=self.update_model_versions,
            inputs=[self.components["model_type"]],
            outputs=[self.components["model_version"]]
        ).then(
            fn=self.update_model_type_and_version,
            inputs=[self.components["model_type"], self.components["model_version"]],
            outputs=[]
        ).then(
            # Update model info and recommended default values based on model and training type
            fn=self.update_model_info,
            inputs=[self.components["model_type"], self.components["training_type"]],
            outputs=[
                self.components["model_info"],
                self.components["train_steps"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"],
                self.components["lora_params_row"],
                self.components["lora_settings_row"],
                self.components["control_params_row"],
                self.components["control_settings_row"],
                self.components["frame_conditioning_row"],
                self.components["control_options_row"],
                self.components["lora_rank"],
                self.components["lora_alpha"]
            ]
        )
        
        # Model version change event
        self.components["model_version"].change(
            fn=self.save_model_version,  # Replace with this new function
            inputs=[self.components["model_type"], self.components["model_version"]],
            outputs=[]
        )
            
        # Training type change event
        self.components["training_type"].change(
            fn=lambda v: self.app.update_ui_state(training_type=v),
            inputs=[self.components["training_type"]],
            outputs=[]
        ).then(
            fn=self.update_model_info,
            inputs=[self.components["model_type"], self.components["training_type"]],
            outputs=[
                self.components["model_info"],
                self.components["train_steps"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"],
                self.components["lora_params_row"],
                self.components["lora_settings_row"],
                self.components["control_params_row"],
                self.components["control_settings_row"],
                self.components["frame_conditioning_row"],
                self.components["control_options_row"],
                self.components["lora_rank"],
                self.components["lora_alpha"]
            ]
        )

        self.components["auto_resume"].change(
            fn=lambda v: self.app.update_ui_state(auto_resume=v),
            inputs=[self.components["auto_resume"]],
            outputs=[]
        )

        # Add in the connect_events() method:
        self.components["num_gpus"].change(
            fn=lambda v: self.app.update_ui_state(num_gpus=v),
            inputs=[self.components["num_gpus"]],
            outputs=[]
        )

        self.components["precomputation_items"].change(
            fn=lambda v: self.app.update_ui_state(precomputation_items=v),
            inputs=[self.components["precomputation_items"]],
            outputs=[]
        )

        self.components["lr_warmup_steps"].change(
            fn=lambda v: self.app.update_ui_state(lr_warmup_steps=v),
            inputs=[self.components["lr_warmup_steps"]],
            outputs=[]
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
        
        # Control parameters change events
        self.components["control_type"].change(
            fn=lambda v: self.app.update_ui_state(control_type=v),
            inputs=[self.components["control_type"]],
            outputs=[]
        )
        
        self.components["train_qk_norm"].change(
            fn=lambda v: self.app.update_ui_state(train_qk_norm=v),
            inputs=[self.components["train_qk_norm"]],
            outputs=[]
        )
        
        self.components["frame_conditioning_type"].change(
            fn=lambda v: self.app.update_ui_state(frame_conditioning_type=v),
            inputs=[self.components["frame_conditioning_type"]],
            outputs=[]
        )
        
        self.components["frame_conditioning_index"].change(
            fn=lambda v: self.app.update_ui_state(frame_conditioning_index=v),
            inputs=[self.components["frame_conditioning_index"]],
            outputs=[]
        )
        
        self.components["frame_conditioning_concatenate_mask"].change(
            fn=lambda v: self.app.update_ui_state(frame_conditioning_concatenate_mask=v),
            inputs=[self.components["frame_conditioning_concatenate_mask"]],
            outputs=[]
        )

        self.components["train_steps"].change(
            fn=lambda v: self.app.update_ui_state(train_steps=v),
            inputs=[self.components["train_steps"]],
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
        
        # Resolution change event
        self.components["resolution"].change(
            fn=lambda v: self.app.update_ui_state(resolution=v),
            inputs=[self.components["resolution"]],
            outputs=[]
        )
        
        # Training control events
        self.components["start_btn"].click(
            fn=self.handle_new_training_start,
            inputs=[
                self.components["model_type"],
                self.components["model_version"],
                self.components["training_type"],
                self.components["lora_rank"],
                self.components["lora_alpha"],
                self.components["train_steps"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"],
                self.app.tabs["manage_tab"].components["repo_id"]
            ],
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.app.tabs["manage_tab"].components["download_model_btn"],
                self.app.tabs["manage_tab"].components["download_checkpoint_btn"]
            ]
        )

        self.components["resume_btn"].click(
            fn=self.handle_resume_training,
            inputs=[
                self.components["model_type"],
                self.components["model_version"],
                self.components["training_type"],
                self.components["lora_rank"],
                self.components["lora_alpha"],
                self.components["train_steps"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"],
                self.app.tabs["manage_tab"].components["repo_id"]
            ],
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.app.tabs["manage_tab"].components["download_model_btn"],
                self.app.tabs["manage_tab"].components["download_checkpoint_btn"]
            ]
        )

        self.components["start_from_lora_btn"].click(
            fn=self.handle_start_from_lora_training,
            inputs=[
                self.components["model_type"],
                self.components["model_version"],
                self.components["training_type"],
                self.components["lora_rank"],
                self.components["lora_alpha"],
                self.components["train_steps"],
                self.components["batch_size"],
                self.components["learning_rate"],
                self.components["save_iterations"],
                self.app.tabs["manage_tab"].components["repo_id"]
            ],
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.app.tabs["manage_tab"].components["download_model_btn"],
                self.app.tabs["manage_tab"].components["download_checkpoint_btn"]
            ]
        )
        

        # Use simplified event handlers for pause/resume and stop
        third_btn = self.components["delete_checkpoints_btn"] if "delete_checkpoints_btn" in self.components else self.components["pause_resume_btn"]
        
        self.components["pause_resume_btn"].click(
            fn=self.handle_pause_resume,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["current_task_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                third_btn,
                self.app.tabs["manage_tab"].components["download_model_btn"],
                self.app.tabs["manage_tab"].components["download_checkpoint_btn"]
            ]
        )

        self.components["stop_btn"].click(
            fn=self.handle_stop,
            outputs=[
                self.components["status_box"],
                self.components["log_box"],
                self.components["current_task_box"],
                self.components["start_btn"],
                self.components["stop_btn"],
                third_btn,
                self.app.tabs["manage_tab"].components["download_model_btn"],
                self.app.tabs["manage_tab"].components["download_checkpoint_btn"]
            ]
        )

        # Add an event handler for delete_checkpoints_btn
        self.components["delete_checkpoints_btn"].click(
            fn=lambda: self.app.training.delete_all_checkpoints(),
            outputs=[self.components["status_box"]]
        )
    
    def update_model_versions(self, model_type: str) -> Dict:
        """Update model version choices based on selected model type"""
        try:
            # Get version choices for this model type
            model_versions = self.get_model_version_choices(model_type)
            
            # Get default version
            default_version = self.get_default_model_version(model_type)
            logger.info(f"update_model_versions({model_type}): default_version = {default_version}, available versions: {model_versions}")
            
            # Update UI state with proper model_type first
            self.app.update_ui_state(model_type=model_type)
            
            # Create a list of tuples (label, value) for the dropdown choices
            # This ensures compatibility with Gradio Dropdown component expectations
            choices_tuples = [(str(version), str(version)) for version in model_versions]
            
            # Create a new dropdown with the updated choices
            if not choices_tuples:
                logger.warning(f"No model versions available for {model_type}, using empty list")
                # Return empty dropdown to avoid errors
                return gr.Dropdown(choices=[], value=None)
                    
            # Ensure default_version is in model_versions
            string_versions = [str(v) for v in model_versions]
            if default_version not in string_versions and string_versions:
                default_version = string_versions[0]
                logger.info(f"Default version not in choices, using first available: {default_version}")
            
            # Return the updated dropdown
            logger.info(f"Returning dropdown with {len(choices_tuples)} choices")
            return gr.Dropdown(choices=choices_tuples, value=default_version)
        except Exception as e:
            # Log any exceptions for debugging
            logger.error(f"Error in update_model_versions: {str(e)}")
            # Return empty dropdown to avoid errors
            return gr.Dropdown(choices=[], value=None)
        
    def handle_training_start(
        self, model_type, model_version, training_type, 
        lora_rank, lora_alpha, train_steps, batch_size, learning_rate, 
        save_iterations, repo_id,
        progress=gr.Progress(),
        resume_from_checkpoint=None,
    ):
        """Handle training start with proper log parser reset and checkpoint detection"""
        
        # Safely reset log parser if it exists
        if hasattr(self.app, 'log_parser') and self.app.log_parser is not None:
            self.app.log_parser.reset()
        else:
            logger.warning("Log parser not initialized, creating a new one")
            from ..utils import TrainingLogParser
            self.app.log_parser = TrainingLogParser()
        
        # Check for latest checkpoint
        checkpoints = list(self.app.output_path.glob("finetrainers_step_*"))
        has_checkpoints = len(checkpoints) > 0
        resume_from = resume_from_checkpoint  # Use the passed parameter
        
        if resume_from and checkpoints:
            # Find the latest checkpoint
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            resume_from = str(latest_checkpoint)

            logger.info(f"Found checkpoint at {resume_from}, note from @julian: right now let's just resume training at 'latest'")
            result_from = "latest"
            
        # Convert model_type display name to internal name
        model_internal_type = MODEL_TYPES.get(model_type)
        
        if not model_internal_type:
            logger.error(f"Invalid model type: {model_type}")
            return f"Error: Invalid model type '{model_type}'", "Model type not recognized"
        
        # Convert training_type display name to internal name
        training_internal_type = TRAINING_TYPES.get(training_type)
        
        if not training_internal_type:
            logger.error(f"Invalid training type: {training_type}")
            return f"Error: Invalid training type '{training_type}'", "Training type not recognized"
        
        # Get other parameters from UI form
        num_gpus = int(self.components["num_gpus"].value)
        precomputation_items = int(self.components["precomputation_items"].value)
        lr_warmup_steps = int(self.components["lr_warmup_steps"].value)
        
        # Get custom prompt prefix from caption tab
        custom_prompt_prefix = None
        if hasattr(self.app, 'tabs') and 'caption_tab' in self.app.tabs:
            caption_tab = self.app.tabs['caption_tab']
            if hasattr(caption_tab, 'components') and 'custom_prompt_prefix' in caption_tab.components:
                custom_prompt_prefix = caption_tab.components['custom_prompt_prefix'].value
        
        # Start training (it will automatically use the checkpoint if provided)
        try:
            return self.app.training.start_training(
                model_internal_type,
                lora_rank,
                lora_alpha,
                train_steps,
                batch_size,
                learning_rate,
                save_iterations,
                repo_id,
                training_type=training_internal_type,
                model_version=model_version,
                resume_from_checkpoint=resume_from,
                num_gpus=num_gpus,
                precomputation_items=precomputation_items,
                lr_warmup_steps=lr_warmup_steps,
                progress=progress,
                custom_prompt_prefix=custom_prompt_prefix
            )
        except Exception as e:
            logger.exception("Error starting training")
            return f"Error starting training: {str(e)}", f"Exception: {str(e)}\n\nCheck the logs for more details."

    def get_model_version_choices(self, model_type: str) -> List[str]:
        """Get model version choices based on model type"""
        # Convert UI display name to internal name
        internal_type = MODEL_TYPES.get(model_type)
        if not internal_type or internal_type not in MODEL_VERSIONS:
            logger.warning(f"No model versions found for {model_type} (internal type: {internal_type})")
            return []
            
        # Return just the model IDs as a list of simple strings
        version_ids = list(MODEL_VERSIONS.get(internal_type, {}).keys())
        logger.info(f"Found {len(version_ids)} versions for {model_type}: {version_ids}")
        
        # Ensure they're all strings
        return [str(version) for version in version_ids]

    def get_default_model_version(self, model_type: str) -> str:
        """Get default model version for the given model type"""
        # Convert UI display name to internal name
        internal_type = MODEL_TYPES.get(model_type)
        logger.debug(f"get_default_model_version({model_type}) = {internal_type}")
        
        if not internal_type or internal_type not in MODEL_VERSIONS:
            logger.warning(f"No valid model versions found for {model_type}")
            return ""
            
        # Get the first version available for this model type
        versions = list(MODEL_VERSIONS.get(internal_type, {}).keys())
        if versions:
            default_version = versions[0]
            logger.debug(f"Default version for {model_type}: {default_version}")
            return default_version
        return ""
   
    def update_model_info(self, model_type: str, training_type: str) -> Dict:
        """Update model info and related UI components based on model type and training type"""
        # Get model info text
        model_info = self.get_model_info(model_type, training_type)
        
        # Add general information about the selected training type
        if training_type == "Full Finetune":
            finetune_info = """
## 🧠 Full Finetune Mode
            
Full finetune mode trains all parameters of the model, requiring more VRAM but potentially enabling higher quality results.
            
- Requires 20-50GB+ VRAM depending on model
- Creates a complete standalone model (~8GB+ file size)
- Recommended only for high-end GPUs (A100, H100, etc.)
- Not recommended for the larger models like Hunyuan Video on consumer hardware
            """
            model_info = finetune_info + "\n\n" + model_info
        
        # Get default parameters for this model type and training type
        params = self.get_default_params(MODEL_TYPES.get(model_type), TRAINING_TYPES.get(training_type))
        
        # Check if LoRA params should be visible
        show_lora_params = training_type in ["LoRA Finetune", "Control LoRA"]
        
        # Check if Control-specific params should be visible
        show_control_params = training_type in ["Control LoRA", "Control Full Finetune"]
        
        # Return updates for UI components
        return {
            self.components["model_info"]: model_info,
            self.components["train_steps"]: params["train_steps"],
            self.components["batch_size"]: params["batch_size"],
            self.components["learning_rate"]: params["learning_rate"],
            self.components["save_iterations"]: params["save_iterations"],
            self.components["lora_rank"]: params["lora_rank"],
            self.components["lora_alpha"]: params["lora_alpha"],
            self.components["lora_params_row"]: gr.Row(visible=show_lora_params),
            self.components["lora_settings_row"]: gr.Row(visible=show_lora_params),
            self.components["control_params_row"]: gr.Row(visible=show_control_params),
            self.components["control_settings_row"]: gr.Row(visible=show_control_params),
            self.components["frame_conditioning_row"]: gr.Row(visible=show_control_params),
            self.components["control_options_row"]: gr.Row(visible=show_control_params)
        }

    def get_model_info(self, model_type: str, training_type: str) -> str:
        """Get information about the selected model type and training method"""
        if model_type == "HunyuanVideo":
            base_info = """## HunyuanVideo Training
- Required VRAM: ~48GB minimum
- Recommended batch size: 1-2
- Typical training time: 2-4 hours
- Default resolution: 49x512x768"""
            
            if training_type == "LoRA Finetune":
                return base_info + "\n- Required VRAM: ~18GB minimum\n- Default LoRA rank: 128 (~400 MB)"
            elif training_type == "Control LoRA":
                return base_info + "\n- Required VRAM: ~20GB minimum\n- Default LoRA rank: 128 (~400 MB)\n- Supports image conditioning"
            elif training_type == "Control Full Finetune":
                return base_info + "\n- Required VRAM: ~50GB minimum\n- Supports image conditioning\n- **Not recommended due to VRAM requirements**"
            else:
                return base_info + "\n- Required VRAM: ~48GB minimum\n- **Full finetune not recommended due to VRAM requirements**"
                
        elif model_type == "LTX-Video":
            base_info = """## LTX-Video Training
- Recommended batch size: 1-4
- Typical training time: 1-3 hours
- Default resolution: 49x512x768"""
            
            if training_type == "LoRA Finetune":
                return base_info + "\n- Required VRAM: ~18GB minimum\n- Default LoRA rank: 128 (~400 MB)"
            elif training_type == "Control LoRA":
                return base_info + "\n- Required VRAM: ~20GB minimum\n- Default LoRA rank: 128 (~400 MB)\n- Supports image conditioning"
            elif training_type == "Control Full Finetune":
                return base_info + "\n- Required VRAM: ~23GB minimum\n- Supports image conditioning"
            else:
                return base_info + "\n- Required VRAM: ~21GB minimum\n- Full model size: ~8GB"
                
        elif model_type == "Wan":
            base_info = """## Wan2.1 Training
- Recommended batch size: 1-4
- Typical training time: 1-3 hours
- Default resolution: 49x512x768"""
            
            if training_type == "LoRA Finetune":
                return base_info + "\n- Required VRAM: ~16GB minimum\n- Default LoRA rank: 32 (~120 MB)"
            elif training_type == "Control LoRA":
                return base_info + "\n- Required VRAM: ~18GB minimum\n- Default LoRA rank: 32 (~120 MB)\n- Supports image conditioning"
            elif training_type == "Control Full Finetune":
                return base_info + "\n- Required VRAM: ~40GB minimum\n- Supports image conditioning\n- **Not recommended due to VRAM requirements**"
            else:
                return base_info + "\n- **Full finetune not recommended due to VRAM requirements**"
        
        # Default fallback
        return f"### {model_type}\nPlease check documentation for VRAM requirements and recommended settings."

    def get_default_params(self, model_type: str, training_type: str) -> Dict[str, Any]:
        """Get default training parameters for model type"""
        # Use model-specific defaults based on model_type and training_type
        model_defaults = {}
        
        if model_type == "hunyuan_video" and training_type in HUNYUAN_VIDEO_DEFAULTS:
            model_defaults = HUNYUAN_VIDEO_DEFAULTS[training_type]
        elif model_type == "ltx_video" and training_type in LTX_VIDEO_DEFAULTS:
            model_defaults = LTX_VIDEO_DEFAULTS[training_type]
        elif model_type == "wan" and training_type in WAN_DEFAULTS:
            model_defaults = WAN_DEFAULTS[training_type]
        
        # Build the complete params dict with defaults plus model-specific overrides
        params = {
            "train_steps": DEFAULT_NB_TRAINING_STEPS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "save_iterations": DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
            "lora_rank": DEFAULT_LORA_RANK_STR,
            "lora_alpha": DEFAULT_LORA_ALPHA_STR
        }
        
        # Override with model-specific values
        params.update(model_defaults)
        
        return params


    def get_latest_status_message_and_logs(self) -> Tuple[str, str, str]:
        """Get latest status message, log content, and status code in a safer way"""
        state = self.app.training.get_status()
        logs = self.app.training.get_logs()

        # Check if training process died unexpectedly
        training_died = False
        
        if state["status"] == "training" and not self.app.training.is_training_running():
            state["status"] = "error"
            state["message"] = "Training process terminated unexpectedly."
            training_died = True
            
            # Look for error in logs
            error_lines = []
            for line in logs.splitlines():
                if "Error:" in line or "Exception:" in line or "Traceback" in line:
                    error_lines.append(line)
            
            if error_lines:
                state["message"] += f"\n\nPossible error: {error_lines[-1]}"

        # Ensure log parser is initialized
        if not hasattr(self.app, 'log_parser') or self.app.log_parser is None:
            from ..utils import TrainingLogParser
            self.app.log_parser = TrainingLogParser()
            logger.info("Initialized missing log parser")

        # Parse new log lines
        if logs and not training_died:
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
        elif "error" in state["message"].lower():
            state["status"] = "error"
        elif "failed" in state["message"].lower():
            state["status"] = "error"
        elif "stopped" in state["message"].lower():
            state["status"] = "stopped"

        # Add the current task info if available
        if hasattr(self.app, 'log_parser') and self.app.log_parser is not None:
            state["current_task"] = self.app.log_parser.get_current_task_display()

        return (state["status"], state["message"], logs)

    def get_status_updates(self):
        """Get status updates for text components (no variant property)"""
        status, message, logs = self.get_latest_status_message_and_logs()
        
        # Get current task if available
        current_task = ""
        if hasattr(self.app, 'log_parser') and self.app.log_parser is not None:
            current_task = self.app.log_parser.get_current_task_display()
        
        return message, logs, current_task

    def get_button_updates(self):
        """Get button updates (with variant property)"""
        status, _, _ = self.get_latest_status_message_and_logs()
        
        # Add checkpoints detection
        checkpoints = list(self.app.output_path.glob("finetrainers_step_*"))
        has_checkpoints = len(checkpoints) > 0
        
        is_training = status in ["training", "initializing"]
        is_completed = status in ["completed", "error", "stopped"]
        
        # Create button updates
        start_btn = gr.Button(
            value="🚀 Start new training",
            interactive=not is_training,
            variant="primary" if not is_training else "secondary"
        )
        
        resume_btn = gr.Button(
            value="🛸 Start from latest checkpoint",
            interactive=has_checkpoints and not is_training,
            variant="primary" if not is_training else "secondary"
        )
        
        stop_btn = gr.Button(
            value="Stop at Last Checkpoint",
            interactive=is_training,
            variant="primary" if is_training else "secondary"
        )
        
        # Add delete_checkpoints_btn
        delete_checkpoints_btn = gr.Button(
            "Delete All Checkpoints",
            interactive=has_checkpoints and not is_training,
            variant="stop"
        )
        
        return start_btn, resume_btn, stop_btn, delete_checkpoints_btn
            
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
        
        # Add current task information to the dedicated box
        if training_state.get("current_task"):
            updates["current_task_box"] = training_state["current_task"]
        else:
            updates["current_task_box"] = "No active task" if training_state["status"] != "training" else "Waiting for task information..."
        
        return updates
        
    def handle_pause_resume(self):
        """Handle pause/resume button click"""
        status, _, _ = self.get_latest_status_message_and_logs()
        
        if status == "paused":
            self.app.training.resume_training()
        else:
            self.app.training.pause_training()
            
        # Return the updates separately for text and buttons
        return (*self.get_status_updates(), *self.get_button_updates())

    def handle_stop(self):
        """Handle stop button click"""
        self.app.training.stop_training()
        
        # Return the updates separately for text and buttons
        return (*self.get_status_updates(), *self.get_button_updates())
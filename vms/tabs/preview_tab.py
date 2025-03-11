"""
Preview tab for Video Model Studio UI 
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from vms.services.base_tab import BaseTab
from vms.config import (
    MODEL_TYPES, DEFAULT_PROMPT_PREFIX
)

logger = logging.getLogger(__name__)

class PreviewTab(BaseTab):
    """Preview tab for testing trained models"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "preview_tab"
        self.title = "6️⃣  Preview"
        
        # Get reference to the preview service from app_state
        self.previewing_service = app_state.previewing
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Preview tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                gr.Markdown("## Test Your Trained Model")
            
            with gr.Row():
                with gr.Column(scale=2):
                    self.components["prompt"] = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    
                    self.components["negative_prompt"] = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter negative prompt here...",
                        lines=3,
                        value="worst quality, low quality, blurry, jittery, distorted, ugly, deformed, disfigured, messy background"
                    )
                    
                    self.components["prompt_prefix"] = gr.Textbox(
                        label="Global Prompt Prefix",
                        placeholder="Prefix to add to all prompts",
                        value=DEFAULT_PROMPT_PREFIX
                    )
                    
                    with gr.Row():
                        self.components["model_type"] = gr.Dropdown(
                            choices=list(MODEL_TYPES.keys()),
                            label="Model Type",
                            value=list(MODEL_TYPES.keys())[0]
                        )
                        
                        self.components["resolution_preset"] = gr.Dropdown(
                            choices=["480p", "720p"],
                            label="Resolution Preset",
                            value="480p"
                        )
                    
                    with gr.Row():
                        self.components["width"] = gr.Number(
                            label="Width",
                            value=832,
                            precision=0
                        )
                        
                        self.components["height"] = gr.Number(
                            label="Height",
                            value=480,
                            precision=0
                        )
                    
                    with gr.Row():
                        self.components["num_frames"] = gr.Slider(
                            label="Number of Frames",
                            minimum=1,
                            maximum=257,
                            step=8,
                            value=49
                        )
                        
                        self.components["fps"] = gr.Slider(
                            label="FPS",
                            minimum=1,
                            maximum=60,
                            step=1,
                            value=16
                        )
                    
                    with gr.Row():
                        self.components["guidance_scale"] = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=5.0
                        )
                        
                        self.components["flow_shift"] = gr.Slider(
                            label="Flow Shift",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=3.0
                        )
                    
                    with gr.Row():
                        self.components["lora_weight"] = gr.Slider(
                            label="LoRA Weight",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.7
                        )
                        
                        self.components["inference_steps"] = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=30
                        )
                    
                    self.components["enable_cpu_offload"] = gr.Checkbox(
                        label="Enable Model CPU Offload (for low-VRAM GPUs)",
                        value=True
                    )
                    
                    self.components["generate_btn"] = gr.Button(
                        "Generate Video",
                        variant="primary"
                    )
                
                with gr.Column(scale=3):
                    self.components["preview_video"] = gr.Video(
                        label="Generated Video",
                        interactive=False
                    )
                    
                    self.components["status"] = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    
                    with gr.Accordion("Log", open=False):
                        self.components["log"] = gr.TextArea(
                            label="Generation Log",
                            interactive=False,
                            lines=10
                        )
        
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Update resolution when preset changes
        self.components["resolution_preset"].change(
            fn=self.update_resolution,
            inputs=[self.components["resolution_preset"]],
            outputs=[
                self.components["width"],
                self.components["height"],
                self.components["flow_shift"]
            ]
        )
        
        # Generate button click
        self.components["generate_btn"].click(
            fn=self.generate_video,
            inputs=[
                self.components["model_type"],
                self.components["prompt"],
                self.components["negative_prompt"],
                self.components["prompt_prefix"],
                self.components["width"],
                self.components["height"],
                self.components["num_frames"],
                self.components["guidance_scale"],
                self.components["flow_shift"],
                self.components["lora_weight"],
                self.components["inference_steps"],
                self.components["enable_cpu_offload"],
                self.components["fps"]
            ],
            outputs=[
                self.components["preview_video"],
                self.components["status"],
                self.components["log"]
            ]
        )
    
    def update_resolution(self, preset: str) -> Tuple[int, int, float]:
        """Update resolution and flow shift based on preset"""
        if preset == "480p":
            return 832, 480, 3.0
        elif preset == "720p":
            return 1280, 720, 5.0
        else:
            return 832, 480, 3.0
    
    def generate_video(
        self,
        model_type: str,
        prompt: str,
        negative_prompt: str,
        prompt_prefix: str,
        width: int,
        height: int,
        num_frames: int,
        guidance_scale: float,
        flow_shift: float,
        lora_weight: float,
        inference_steps: int,
        enable_cpu_offload: bool,
        fps: int
    ) -> Tuple[Optional[str], str, str]:
        """Handler for generate button click, delegates to preview service"""
        return self.preview_service.generate_video(
            model_type=model_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_prefix=prompt_prefix,
            width=width,
            height=height,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            lora_weight=lora_weight,
            inference_steps=inference_steps,
            enable_cpu_offload=enable_cpu_offload,
            fps=fps
        )
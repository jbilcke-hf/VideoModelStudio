"""
Preview tab for Video Model Studio UI 
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time

from .base_tab import BaseTab
from ..config import (
    MODEL_TYPES, DEFAULT_PROMPT_PREFIX
)

logger = logging.getLogger(__name__)

class PreviewTab(BaseTab):
    """Preview tab for testing trained models"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "preview_tab"
        self.title = "6️⃣  Preview"
         
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
                        # Get the currently selected model type from training tab if possible
                        default_model = self.get_default_model_type()
                        
                        # Make model_type read-only (disabled), as it must match what was trained
                        self.components["model_type"] = gr.Dropdown(
                            choices=list(MODEL_TYPES.keys()),
                            label="Model Type (from training)",
                            value=default_model,
                            interactive=False
                        )
                        
                        # Add model variant selection based on model type
                        self.components["model_variant"] = gr.Dropdown(
                            label="Model Variant",
                            choices=self.get_variant_choices(default_model),
                            value=self.get_default_variant(default_model)
                        )
                    
                    # Add image input for image-to-video models
                    self.components["conditioning_image"] = gr.Image(
                        label="Conditioning Image (for Image-to-Video models)",
                        type="filepath",
                        visible=False
                    )
                    
                    with gr.Row():
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
                    
                    with gr.Accordion("Log", open=True):
                        self.components["log"] = gr.TextArea(
                            label="Generation Log",
                            interactive=False,
                            lines=15
                        )
        
        return tab
    
    def get_variant_choices(self, model_type: str) -> List[str]:
        """Get model variant choices based on model type"""
        # Convert UI display name to internal name
        internal_type = MODEL_TYPES.get(model_type)
        if not internal_type:
            return []
            
        # Get variants from preview service
        variants = self.app.previewing.get_model_variants(internal_type)
        if not variants:
            return []
            
        # Format choices with display name and description
        choices = []
        for model_id, info in variants.items():
            choices.append(f"{model_id} - {info.get('name', '')}")
            
        return choices
    
    def get_default_variant(self, model_type: str) -> str:
        """Get default model variant for the model type"""
        choices = self.get_variant_choices(model_type)
        if choices:
            return choices[0]
        return ""
    
    def get_default_model_type(self) -> str:
        """Get the currently selected model type from training tab"""
        try:
            # Try to get the model type from UI state
            ui_state = self.app.training.load_ui_state()
            model_type = ui_state.get("model_type")
            
            # Make sure it's a valid model type
            if model_type in MODEL_TYPES:
                return model_type
            
            # If we couldn't get a valid model type, try to get it from the training tab directly
            if hasattr(self.app, 'tabs') and 'train_tab' in self.app.tabs:
                train_tab = self.app.tabs['train_tab']
                if hasattr(train_tab, 'components') and 'model_type' in train_tab.components:
                    train_model_type = train_tab.components['model_type'].value
                    if train_model_type in MODEL_TYPES:
                        return train_model_type
            
            # Fallback to first model type
            return list(MODEL_TYPES.keys())[0]
        except Exception as e:
            logger.warning(f"Failed to get default model type: {e}")
            return list(MODEL_TYPES.keys())[0]
    
    def extract_model_id(self, variant_choice: str) -> str:
        """Extract model ID from variant choice string"""
        if " - " in variant_choice:
            return variant_choice.split(" - ")[0].strip()
        return variant_choice
    
    def get_variant_type(self, model_type: str, model_variant: str) -> str:
        """Get the variant type (text-to-video or image-to-video)"""
        # Convert UI display name to internal name
        internal_type = MODEL_TYPES.get(model_type)
        if not internal_type:
            return "text-to-video"
            
        # Extract model_id from variant choice
        model_id = self.extract_model_id(model_variant)
            
        # Get variants from preview service
        variants = self.app.previewing.get_model_variants(internal_type)
        variant_info = variants.get(model_id, {})
        
        # Return the variant type or default to text-to-video
        return variant_info.get("type", "text-to-video")
    
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
        
        # Update model_variant choices when model_type changes or tab is selected
        if hasattr(self.app, 'tabs_component') and self.app.tabs_component is not None:
            self.app.tabs_component.select(
                fn=self.sync_model_type_and_variants,
                inputs=[],
                outputs=[
                    self.components["model_type"],
                    self.components["model_variant"]
                ]
            )
        
        # Update variant-specific UI elements when variant changes
        self.components["model_variant"].change(
            fn=self.update_variant_ui,
            inputs=[
                self.components["model_type"],
                self.components["model_variant"]
            ],
            outputs=[
                self.components["conditioning_image"]
            ]
        )
        
        # Load preview UI state when the tab is selected
        if hasattr(self.app, 'tabs_component') and self.app.tabs_component is not None:
            self.app.tabs_component.select(
                fn=self.load_preview_state,
                inputs=[],
                outputs=[
                    self.components["prompt"],
                    self.components["negative_prompt"],
                    self.components["prompt_prefix"],
                    self.components["width"],
                    self.components["height"],
                    self.components["num_frames"],
                    self.components["fps"],
                    self.components["guidance_scale"],
                    self.components["flow_shift"],
                    self.components["lora_weight"],
                    self.components["inference_steps"],
                    self.components["enable_cpu_offload"],
                    self.components["model_variant"]
                ]
            )
        
        # Save preview UI state when values change
        for component_name in [
            "prompt", "negative_prompt", "prompt_prefix", "model_variant", "resolution_preset",
            "width", "height", "num_frames", "fps", "guidance_scale", "flow_shift",
            "lora_weight", "inference_steps", "enable_cpu_offload"
        ]:
            if component_name in self.components:
                self.components[component_name].change(
                    fn=self.save_preview_state_value,
                    inputs=[self.components[component_name]],
                    outputs=[]
                )
        
        # Generate button click
        self.components["generate_btn"].click(
            fn=self.generate_video,
            inputs=[
                self.components["model_type"],
                self.components["model_variant"],
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
                self.components["fps"],
                self.components["conditioning_image"]
            ],
            outputs=[
                self.components["preview_video"],
                self.components["status"],
                self.components["log"]
            ]
        )
    
    def update_variant_ui(self, model_type: str, model_variant: str) -> Dict[str, Any]:
        """Update UI based on the selected model variant"""
        variant_type = self.get_variant_type(model_type, model_variant)
        
        # Show conditioning image input only for image-to-video models
        show_conditioning_image = variant_type == "image-to-video"
        
        return {
            self.components["conditioning_image"]: gr.Image(visible=show_conditioning_image)
        }
    
    def sync_model_type_and_variants(self) -> Tuple[str, str]:
        """Sync model type with training tab when preview tab is selected and update variant choices"""
        model_type = self.get_default_model_type()
        model_variant = self.get_default_variant(model_type)
        return model_type, model_variant
    
    def update_resolution(self, preset: str) -> Tuple[int, int, float]:
        """Update resolution and flow shift based on preset"""
        if preset == "480p":
            return 832, 480, 3.0
        elif preset == "720p":
            return 1280, 720, 5.0
        else:
            return 832, 480, 3.0
    
    def load_preview_state(self) -> Tuple:
        """Load saved preview UI state"""
        # Try to get the saved state
        try:
            state = self.app.training.load_ui_state()
            preview_state = state.get("preview", {})
            
            # Get model type (can't be changed in UI)
            model_type = self.get_default_model_type()
            
            # If model_variant not in choices for current model_type, use default
            model_variant = preview_state.get("model_variant", "")
            variant_choices = self.get_variant_choices(model_type)
            if model_variant not in variant_choices and variant_choices:
                model_variant = variant_choices[0]
            
            return (
                preview_state.get("prompt", ""),
                preview_state.get("negative_prompt", "worst quality, low quality, blurry, jittery, distorted, ugly, deformed, disfigured, messy background"),
                preview_state.get("prompt_prefix", DEFAULT_PROMPT_PREFIX),
                preview_state.get("width", 832),
                preview_state.get("height", 480),
                preview_state.get("num_frames", 49),
                preview_state.get("fps", 16),
                preview_state.get("guidance_scale", 5.0),
                preview_state.get("flow_shift", 3.0),
                preview_state.get("lora_weight", 0.7),
                preview_state.get("inference_steps", 30),
                preview_state.get("enable_cpu_offload", True),
                model_variant
            )
        except Exception as e:
            logger.error(f"Error loading preview state: {e}")
            # Return defaults if loading fails
            return (
                "", 
                "worst quality, low quality, blurry, jittery, distorted, ugly, deformed, disfigured, messy background", 
                DEFAULT_PROMPT_PREFIX,
                832, 480, 49, 16, 5.0, 3.0, 0.7, 30, True,
                self.get_default_variant(self.get_default_model_type())
            )
    
    def save_preview_state_value(self, value: Any) -> None:
        """Save an individual preview state value"""
        try:
            # Get the component name from the event context
            import inspect
            frame = inspect.currentframe()
            frame = inspect.getouterframes(frame)[1]
            event_context = frame.frame.f_locals
            component = event_context.get('component')
            
            if component is None:
                return
            
            # Find the component name
            component_name = None
            for name, comp in self.components.items():
                if comp == component:
                    component_name = name
                    break
            
            if component_name is None:
                return
            
            # Load current state
            state = self.app.training.load_ui_state()
            if "preview" not in state:
                state["preview"] = {}
            
            # Update the value
            state["preview"][component_name] = value
            
            # Save state
            self.app.training.save_ui_state(state)
        except Exception as e:
            logger.error(f"Error saving preview state: {e}")
    
    def generate_video(
        self,
        model_type: str,
        model_variant: str,
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
        fps: int,
        conditioning_image: Optional[str] = None
    ) -> Tuple[Optional[str], str, str]:
        """Handler for generate button click, delegates to preview service"""
        # Save all the parameters to preview state before generating
        try:
            state = self.app.training.load_ui_state()
            if "preview" not in state:
                state["preview"] = {}
                
            # Extract model ID from variant choice
            model_variant_id = self.extract_model_id(model_variant)
                
            # Update all values
            preview_state = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "prompt_prefix": prompt_prefix,
                "model_type": model_type,
                "model_variant": model_variant,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "guidance_scale": guidance_scale,
                "flow_shift": flow_shift,
                "lora_weight": lora_weight,
                "inference_steps": inference_steps,
                "enable_cpu_offload": enable_cpu_offload
            }
            
            state["preview"] = preview_state
            self.app.training.save_ui_state(state)
        except Exception as e:
            logger.error(f"Error saving preview state before generation: {e}")
        
        # Clear the log display at the start to make room for new logs
        # Yield and sleep briefly to allow UI update
        yield None, "Starting generation...", ""
        time.sleep(0.1)
        
        # Extract model ID from variant choice string
        model_variant_id = self.extract_model_id(model_variant)
        
        # Use streaming updates to provide real-time feedback during generation
        def generate_with_updates():
            # Initial UI update
            yield None, "Initializing generation...", "Starting video generation process..."
            
            # Start actual generation
            result = self.app.previewing.generate_video(
                model_type=model_type,
                model_variant=model_variant_id,
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
                fps=fps,
                conditioning_image=conditioning_image
            )
            
            # Return final result
            return result
        
        # Return the generator for streaming updates
        return generate_with_updates()
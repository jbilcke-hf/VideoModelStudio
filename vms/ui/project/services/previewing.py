"""
Preview service for Video Model Studio

Handles the video generation logic and model integration
"""

import logging
import tempfile
import traceback
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import time

from vms.config import (
    STORAGE_PATH, MODEL_TYPES,
    DEFAULT_PROMPT_PREFIX, MODEL_VERSIONS
)
from vms.utils import format_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PreviewingService:
    """Handles the video generation logic and model integration"""
    
    def __init__(self, app=None):
        """Initialize the preview service
        
        Args:
            app: Reference to main application
        """
        self.app = app
    
    def find_latest_lora_weights(self) -> Optional[str]:
        """Find the latest LoRA weights file"""
        try:
            # Check if the root level file exists (this should be the primary location)
            lora_path = self.app.output_path / "pytorch_lora_weights.safetensors"
            if lora_path.exists():
                return str(lora_path)
            
            # Check in lora_weights directory
            lora_weights_dir = self.app.output_path / "lora_weights"
            if lora_weights_dir.exists():
                # Look for the latest checkpoint directory in lora_weights
                lora_checkpoints = [d for d in lora_weights_dir.glob("*") if d.is_dir() and d.name.isdigit()]
                if lora_checkpoints:
                    latest_lora_checkpoint = max(lora_checkpoints, key=lambda x: int(x.name))
                    
                    # Check for weights in the latest LoRA checkpoint
                    possible_weight_files = [
                        "pytorch_lora_weights.safetensors",
                        "adapter_model.safetensors", 
                        "pytorch_model.safetensors",
                        "model.safetensors"
                    ]
                    
                    for weight_file in possible_weight_files:
                        weight_path = latest_lora_checkpoint / weight_file
                        if weight_path.exists():
                            return str(weight_path)
                    
                    # Check if any .safetensors files exist
                    safetensors_files = list(latest_lora_checkpoint.glob("*.safetensors"))
                    if safetensors_files:
                        return str(safetensors_files[0])
            
            # If not found in lora_weights, try to find in finetrainers checkpoints
            checkpoints = list(self.app.output_path.glob("finetrainers_step_*"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("_")[-1]))
                lora_path = latest_checkpoint / "pytorch_lora_weights.safetensors"
                if lora_path.exists():
                    return str(lora_path)
            
            return None
        except Exception as e:
            logger.error(f"Error finding LoRA weights: {e}")
            return None
    
    def get_model_versions(self, model_type: str) -> Dict[str, Dict[str, str]]:
        """Get available model versions for the given model type"""
        return MODEL_VERSIONS.get(model_type, {})
    
    def generate_video(
        self,
        model_type: str,
        model_version: str,
        prompt: str,
        negative_prompt: str,
        prompt_prefix: str,
        width: int,
        height: int,
        num_frames: int,
        guidance_scale: float,
        flow_shift: float,
        lora_scale: float,
        inference_steps: int,
        seed: int = -1,
        enable_cpu_offload: bool = True,
        fps: int = 16,
        first_frame_image: Optional[str] = None,
        last_frame_image: Optional[str] = None
    ) -> Tuple[Optional[str], str, str]:
        """Generate a video using the trained model"""
        try:
            log_messages = []
            print("generate_video")
            
            def log(msg: str):
                log_messages.append(msg)
                logger.info(msg)
                # Return updated log string for UI updates
                return "\n".join(log_messages)
            
            # Find latest LoRA weights if lora_scale > 0
            lora_path = None
            using_lora = lora_scale > 0
            
            if using_lora:
                lora_path = self.find_latest_lora_weights()
                if not lora_path:
                    return None, "Error: No LoRA weights found", log("Error: No LoRA weights found in output directory")
                log(f"Using LoRA weights with scale {lora_scale}")
            else:
                log("Using original model without LoRA weights")
            
            # Add prefix to prompt
            if prompt_prefix and not prompt.startswith(prompt_prefix):
                full_prompt = f"{prompt_prefix}{prompt}"
            else:
                full_prompt = prompt
            
            # Create correct num_frames (should be 8*k + 1)
            adjusted_num_frames = ((num_frames - 1) // 8) * 8 + 1
            if adjusted_num_frames != num_frames:
                log(f"Adjusted number of frames from {num_frames} to {adjusted_num_frames} to match model requirements")
                num_frames = adjusted_num_frames
            
            # Get model type (internal name)
            internal_model_type = MODEL_TYPES.get(model_type)
            if not internal_model_type:
                return None, f"Error: Invalid model type {model_type}", log(f"Error: Invalid model type {model_type}")
            
            # Check if model version is valid
            # This section uses model_version directly from parameter
            if model_version:
                # Verify that the specified model_version exists in our versions
                versions = self.get_model_versions(internal_model_type)
                if model_version not in versions:
                    log(f"Warning: Specified model version '{model_version}' is not recognized")
                    # Fall back to default version for this model
                    if len(versions) > 0:
                        model_version = next(iter(versions.keys()))
                        log(f"Using default model version instead: {model_version}")
                else:
                    log(f"Using specified model version: {model_version}")
            else:
                # No model version specified, use default
                versions = self.get_model_versions(internal_model_type)
                if len(versions) > 0:
                    model_version = next(iter(versions.keys()))
                    log(f"No model version specified, using default: {model_version}")
                else:
                    # Fall back to hardcoded defaults if no versions defined
                    if internal_model_type == "wan":
                        model_version = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                    elif internal_model_type == "ltx_video":
                        model_version = "Lightricks/LTX-Video"
                    elif internal_model_type == "hunyuan_video":
                        model_version = "hunyuanvideo-community/HunyuanVideo"
                    log(f"No versions defined for model type, using default: {model_version}")
            
            # Check if this is an image-to-video or frame-to-video model but no image was provided
            model_version_info = versions.get(model_version, {})
            if model_version_info.get("type") in ["image-to-video", "frame-to-video"] and not first_frame_image:
                model_type_name = "frame conditioning" if model_version_info.get("type") == "frame-to-video" else "conditioning"
                return None, f"Error: This model requires a {model_type_name} image", log(f"Error: This model version requires a {model_type_name} image but none was provided")
            
            # Additional check for FLF2V models that require both first and last frame
            if model_version_info.get("type") == "frame-to-video" and first_frame_image and not last_frame_image:
                return None, "Error: FLF2V models require both first and last frame images", log("Error: FLF2V models require both first and last frame images but only first frame was provided")
            
            log(f"Generating video with model type: {internal_model_type}")
            log(f"Using model version: {model_version}")
            if using_lora and lora_path:
                log(f"Using LoRA weights from: {lora_path}")
            log(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {fps}")
            log(f"Guidance Scale: {guidance_scale}, Flow Shift: {flow_shift}, LoRA Scale: {lora_scale if using_lora else 0}")
            log(f"Generation Seed: {seed}")
            #log(f"Prompt: {full_prompt}")
            #log(f"Negative Prompt: {negative_prompt}")
            
            # Import required components based on model type
            if internal_model_type == "wan":
                return self.generate_wan_video(
                    full_prompt, negative_prompt, width, height, num_frames,
                    guidance_scale, flow_shift, lora_path, lora_scale,
                    inference_steps, seed, enable_cpu_offload, fps, log,
                    model_version, first_frame_image, last_frame_image
                )
            elif internal_model_type == "ltx_video":
                return self.generate_ltx_video(
                    full_prompt, negative_prompt, width, height, num_frames,
                    guidance_scale, flow_shift, lora_path, lora_scale,
                    inference_steps, seed, enable_cpu_offload, fps, log,
                    model_version, conditioning_image
                )
            elif internal_model_type == "hunyuan_video":
                return self.generate_hunyuan_video(
                    full_prompt, negative_prompt, width, height, num_frames,
                    guidance_scale, flow_shift, lora_path, lora_scale,
                    inference_steps, seed, enable_cpu_offload, fps, log,
                    model_version, conditioning_image
                )
            else:
                return None, f"Error: Unsupported model type {internal_model_type}", log(f"Error: Unsupported model type {internal_model_type}")
        
        except Exception as e:
            logger.exception("Error generating video")
            return None, f"Error: {str(e)}", f"Exception occurred: {str(e)}"
    
    def generate_wan_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        guidance_scale: float,
        flow_shift: float,
        lora_path: str,
        lora_scale: float,
        inference_steps: int,
        seed: int = -1,
        enable_cpu_offload: bool = True,
        fps: int = 16,
        log_fn: Callable = print,
        model_version: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        first_frame_image: Optional[str] = None,
        last_frame_image: Optional[str] = None
    ) -> Tuple[Optional[str], str, str]:
        """Generate video using Wan model"""

        try:
            import torch
            import numpy as np
            from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            from diffusers.utils import export_to_video
            from transformers import CLIPVisionModel
            from PIL import Image
            import os

            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
        
            print("Initializing wan generation..")
            log_fn("Importing Wan model components...")
            
            # Set up random seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
                log_fn(f"Using randomly generated seed: {seed}")
            
            # Set random seeds for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda")
            generator = generator.manual_seed(seed)
            
            # Check if this is an I2V or FLF2V model that requires WanImageToVideoPipeline
            is_i2v = "I2V" in model_version
            is_flf2v = "FLF2V" in model_version
            uses_image_pipeline = is_i2v or is_flf2v
            
            log_fn(f"Loading VAE from {model_version}...")
            vae = AutoencoderKLWan.from_pretrained(model_version, subfolder="vae", torch_dtype=torch.float32)
            
            if uses_image_pipeline:
                model_type_str = "FLF2V" if is_flf2v else "I2V"
                log_fn(f"Loading image encoder for {model_type_str} model from {model_version}...")
                image_encoder = CLIPVisionModel.from_pretrained(model_version, subfolder="image_encoder", torch_dtype=torch.float32)
                
                log_fn(f"Loading WanImageToVideoPipeline from {model_version}...")
                pipe = WanImageToVideoPipeline.from_pretrained(
                    model_version, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
                )
            else:
                log_fn(f"Loading WanPipeline from {model_version}...")
                pipe = WanPipeline.from_pretrained(model_version, vae=vae, torch_dtype=torch.bfloat16)
                
                log_fn(f"Configuring scheduler with flow_shift={flow_shift}...")
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, 
                    flow_shift=flow_shift
                )
            
            log_fn("Moving pipeline to CUDA device...")
            pipe.to("cuda")
            
            if enable_cpu_offload:
                log_fn("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload()
                
            # Apply LoRA weights if using them
            if lora_scale > 0 and lora_path:
                log_fn(f"Loading LoRA weights from {lora_path} with lora scale {lora_scale}...")
                pipe.load_lora_weights(lora_path)
            else:
                log_fn("Using base model without LoRA weights")
                
            # Create temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                output_path = temp_file.name
            
            log_fn("Starting video generation...")
            start_time.record()
            
            # Generate video based on model type
            
            if (is_i2v or is_flf2v) and first_frame_image:
                log_fn(f"Loading first frame image from {first_frame_image}...")
                first_frame = Image.open(first_frame_image).convert("RGB")
                first_frame = first_frame.resize((width, height))
                
                if is_flf2v:
                    log_fn("Generating video with frame conditioning (FLF2V)...")
                    if not last_frame_image:
                        return None, "Error: FLF2V model requires both first and last frame images", log_fn("Error: FLF2V model requires both first and last frame images")
                    
                    log_fn(f"Loading last frame image from {last_frame_image}...")
                    last_frame = Image.open(last_frame_image).convert("RGB")
                    last_frame = last_frame.resize((width, height))
                    
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=first_frame,
                        last_image=last_frame,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=inference_steps,
                        cross_attention_kwargs={"scale": lora_scale},
                        generator=generator,
                    ).frames[0]
                else:
                    log_fn("Generating video with image conditioning...")
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=first_frame,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=inference_steps,
                        cross_attention_kwargs={"scale": lora_scale},
                        generator=generator,
                    ).frames[0]
            else:
                log_fn("Generating video with text-only conditioning...")
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=inference_steps,
                    generator=generator,
                ).frames[0]
            
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            
            log_fn(f"Video generation completed in {format_time(generation_time)}")
            log_fn(f"Exporting video to {output_path}...")
            
            export_to_video(output, output_path, fps=fps)
            
            log_fn("Video generation and export completed successfully!")
            
            # Clean up CUDA memory
            pipe = None
            torch.cuda.empty_cache()
            
            return output_path, "Video generated successfully!", log_fn(f"Generation completed in {format_time(generation_time)}")
        
        except Exception as e:
            traceback.print_exc()
            log_fn(f"Error generating video with Wan: {str(e)}")
            # Clean up CUDA memory
            torch.cuda.empty_cache()
            return None, f"Error: {str(e)}", log_fn(f"Exception occurred: {str(e)}")
            
    def generate_ltx_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        guidance_scale: float,
        flow_shift: float,
        lora_path: str,
        lora_scale: float,
        inference_steps: int,
        seed: int = -1,
        enable_cpu_offload: bool = True,
        fps: int = 16,
        log_fn: Callable = print,
        model_version: str = "Lightricks/LTX-Video",
        conditioning_image: Optional[str] = None
    ) -> Tuple[Optional[str], str, str]:
        """Generate video using LTX model"""

        try:
            import torch
            import numpy as np
            from diffusers import LTXPipeline
            from diffusers.utils import export_to_video
            from PIL import Image
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            # Set up random seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
                log_fn(f"Using randomly generated seed: {seed}")
            
            # Set random seeds for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda")
            generator = generator.manual_seed(seed)
        
            log_fn("Importing LTX model components...")
            
            log_fn(f"Loading pipeline from {model_version}...")
            pipe = LTXPipeline.from_pretrained(model_version, torch_dtype=torch.bfloat16)
            
            log_fn("Moving pipeline to CUDA device...")
            pipe.to("cuda")
            
            if enable_cpu_offload:
                log_fn("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload()
            
            # Apply LoRA weights if using them
            if lora_scale > 0 and lora_path:
                log_fn(f"Loading LoRA weights from {lora_path} with lora scale {lora_scale}...")
                pipe.load_lora_weights(lora_path)
            else:
                log_fn("Using base model without LoRA weights")
            
            # Create temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                output_path = temp_file.name
            
            log_fn("Starting video generation...")
            start_time.record()
            
            # LTX doesn't currently support image conditioning in the standard way
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                num_inference_steps=inference_steps,
                cross_attention_kwargs={"scale": lora_scale},
                generator=generator,
            ).frames[0]
            
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            
            log_fn(f"Video generation completed in {format_time(generation_time)}")
            log_fn(f"Exporting video to {output_path}...")
            
            export_to_video(video, output_path, fps=fps)
            
            log_fn("Video generation and export completed successfully!")
            
            # Clean up CUDA memory
            pipe = None
            torch.cuda.empty_cache()
            
            return output_path, "Video generated successfully!", log_fn(f"Generation completed in {format_time(generation_time)}")
        
        except Exception as e:
            log_fn(f"Error generating video with LTX: {str(e)}")
            # Clean up CUDA memory
            torch.cuda.empty_cache()
            return None, f"Error: {str(e)}", log_fn(f"Exception occurred: {str(e)}")
    
    def generate_hunyuan_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        guidance_scale: float,
        flow_shift: float,
        lora_path: str,
        lora_scale: float,
        inference_steps: int,
        seed: int = -1,
        enable_cpu_offload: bool = True,
        fps: int = 16,
        log_fn: Callable = print,
        model_version: str = "hunyuanvideo-community/HunyuanVideo",
        conditioning_image: Optional[str] = None
    ) -> Tuple[Optional[str], str, str]:
        """Generate video using HunyuanVideo model"""

        
        try:
            import torch
            import numpy as np
            from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo
            from diffusers.utils import export_to_video
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            # Set up random seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
                log_fn(f"Using randomly generated seed: {seed}")
            
            # Set random seeds for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda")
            generator = generator.manual_seed(seed)

            log_fn("Importing HunyuanVideo model components...")
            
            log_fn(f"Loading transformer from {model_version}...")
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                model_version, 
                subfolder="transformer", 
                torch_dtype=torch.bfloat16
            )
            
            log_fn(f"Loading pipeline from {model_version}...")
            pipe = HunyuanVideoPipeline.from_pretrained(
                model_version, 
                transformer=transformer,
                torch_dtype=torch.float16
            )
            
            log_fn("Enabling VAE tiling for better memory usage...")
            pipe.vae.enable_tiling()
            
            log_fn("Moving pipeline to CUDA device...")
            pipe.to("cuda")
            
            if enable_cpu_offload:
                log_fn("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload()
            
            # Apply LoRA weights if using them
            if lora_scale > 0 and lora_path:
                log_fn(f"Loading LoRA weights from {lora_path} with lora scale {lora_scale}...")
                pipe.load_lora_weights(lora_path)
            else:
                log_fn("Using base model without LoRA weights")
            
            # Create temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                output_path = temp_file.name
            
            log_fn("Starting video generation...")
            start_time.record()
            
            # Make sure negative_prompt is a list or None
            neg_prompt = [negative_prompt] if negative_prompt else None
            
            output = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                true_cfg_scale=1.0,
                num_inference_steps=inference_steps,
                cross_attention_kwargs={"scale": lora_scale},
                generator=generator,
            ).frames[0]
            
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            
            log_fn(f"Video generation completed in {format_time(generation_time)}")
            log_fn(f"Exporting video to {output_path}...")
            
            export_to_video(output, output_path, fps=fps)
            
            log_fn("Video generation and export completed successfully!")
            
            # Clean up CUDA memory
            pipe = None
            torch.cuda.empty_cache()
            
            return output_path, "Video generated successfully!", log_fn(f"Generation completed in {format_time(generation_time)}")
        
        except Exception as e:
            log_fn(f"Error generating video with HunyuanVideo: {str(e)}")
            # Clean up CUDA memory
            torch.cuda.empty_cache()
            return None, f"Error: {str(e)}", log_fn(f"Exception occurred: {str(e)}")
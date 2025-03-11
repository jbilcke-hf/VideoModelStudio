"""
Preview service for Video Model Studio

Handles the video generation logic and model integration
"""

import logging
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

from vms.config import (
    OUTPUT_PATH, STORAGE_PATH, MODEL_TYPES, TRAINING_PATH,
    DEFAULT_PROMPT_PREFIX
)
from vms.utils import format_time

logger = logging.getLogger(__name__)

class PreviewingService:
    """Handles the video generation logic and model integration"""
    
    def __init__(self):
        """Initialize the preview service"""
        pass
    
    def find_latest_lora_weights(self) -> Optional[str]:
        """Find the latest LoRA weights file"""
        try:
            lora_path = OUTPUT_PATH / "pytorch_lora_weights.safetensors"
            if lora_path.exists():
                return str(lora_path)
            
            # If not found in the expected location, try to find in checkpoints
            checkpoints = list(OUTPUT_PATH.glob("checkpoint-*"))
            if not checkpoints:
                return None
            
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            lora_path = latest_checkpoint / "pytorch_lora_weights.safetensors"
            
            if lora_path.exists():
                return str(lora_path)
            
            return None
        except Exception as e:
            logger.error(f"Error finding LoRA weights: {e}")
            return None
    
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
        """Generate a video using the trained model"""
        try:
            log_messages = []
            
            def log(msg: str):
                log_messages.append(msg)
                logger.info(msg)
                return "\n".join(log_messages)
            
            # Find latest LoRA weights
            lora_path = self.find_latest_lora_weights()
            if not lora_path:
                return None, "Error: No LoRA weights found", log("Error: No LoRA weights found in output directory")
            
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
            
            log(f"Generating video with model type: {internal_model_type}")
            log(f"Using LoRA weights from: {lora_path}")
            log(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {fps}")
            log(f"Guidance Scale: {guidance_scale}, Flow Shift: {flow_shift}, LoRA Weight: {lora_weight}")
            log(f"Prompt: {full_prompt}")
            log(f"Negative Prompt: {negative_prompt}")
            
            # Import required components based on model type
            if internal_model_type == "wan":
                return self.generate_wan_video(
                    full_prompt, negative_prompt, width, height, num_frames,
                    guidance_scale, flow_shift, lora_path, lora_weight,
                    inference_steps, enable_cpu_offload, fps, log
                )
            elif internal_model_type == "ltx_video":
                return self.generate_ltx_video(
                    full_prompt, negative_prompt, width, height, num_frames,
                    guidance_scale, flow_shift, lora_path, lora_weight,
                    inference_steps, enable_cpu_offload, fps, log
                )
            elif internal_model_type == "hunyuan_video":
                return self.generate_hunyuan_video(
                    full_prompt, negative_prompt, width, height, num_frames,
                    guidance_scale, flow_shift, lora_path, lora_weight,
                    inference_steps, enable_cpu_offload, fps, log
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
        lora_weight: float,
        inference_steps: int,
        enable_cpu_offload: bool,
        fps: int,
        log_fn: Callable
    ) -> Tuple[Optional[str], str, str]:
        """Generate video using Wan model"""
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            from diffusers.utils import export_to_video
            
            log_fn("Importing Wan model components...")
            
            # Use the smaller model for faster inference
            model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            
            log_fn(f"Loading VAE from {model_id}...")
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            
            log_fn(f"Loading transformer from {model_id}...")
            pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            
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
            
            log_fn(f"Loading LoRA weights from {lora_path} with weight {lora_weight}...")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_weight)
            
            # Create temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                output_path = temp_file.name
            
            log_fn("Starting video generation...")
            start_time.record()
            
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
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
        lora_weight: float,
        inference_steps: int,
        enable_cpu_offload: bool,
        fps: int,
        log_fn: Callable
    ) -> Tuple[Optional[str], str, str]:
        """Generate video using LTX model"""
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            import torch
            from diffusers import LTXPipeline
            from diffusers.utils import export_to_video
            
            log_fn("Importing LTX model components...")
            
            model_id = "Lightricks/LTX-Video"
            
            log_fn(f"Loading pipeline from {model_id}...")
            pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            
            log_fn("Moving pipeline to CUDA device...")
            pipe.to("cuda")
            
            if enable_cpu_offload:
                log_fn("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload()
            
            log_fn(f"Loading LoRA weights from {lora_path} with weight {lora_weight}...")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_weight)
            
            # Create temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                output_path = temp_file.name
            
            log_fn("Starting video generation...")
            start_time.record()
            
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
        lora_weight: float,
        inference_steps: int,
        enable_cpu_offload: bool,
        fps: int,
        log_fn: Callable
    ) -> Tuple[Optional[str], str, str]:
        """Generate video using HunyuanVideo model"""
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            import torch
            from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, AutoencoderKLHunyuanVideo
            from diffusers.utils import export_to_video
            
            log_fn("Importing HunyuanVideo model components...")
            
            model_id = "hunyuanvideo-community/HunyuanVideo"
            
            log_fn(f"Loading transformer from {model_id}...")
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                model_id, 
                subfolder="transformer", 
                torch_dtype=torch.bfloat16
            )
            
            log_fn(f"Loading pipeline from {model_id}...")
            pipe = HunyuanVideoPipeline.from_pretrained(
                model_id, 
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
            
            log_fn(f"Loading LoRA weights from {lora_path} with weight {lora_weight}...")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_weight)
            
            # Create temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                output_path = temp_file.name
            
            log_fn("Starting video generation...")
            start_time.record()
            
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                true_cfg_scale=1.0,
                num_inference_steps=inference_steps,
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
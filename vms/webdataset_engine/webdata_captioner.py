"""
Captioning service for WebDataset
"""

import os
import io
import tempfile
import asyncio
import logging
import uuid
import copy
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
import gradio as gr
import webdataset as wds
from PIL import Image
from decord import VideoReader, cpu

from .webdataset_manager import WebDatasetManager
from .shard_writer import ShardWriter

from ..config import (
    STORAGE_PATH, 
    PRELOAD_CAPTIONING_MODEL, 
    CAPTIONING_MODEL,
    USE_MOCK_CAPTIONING_MODEL,
    DEFAULT_CAPTIONING_BOT_INSTRUCTIONS,
    DEFAULT_PROMPT_PREFIX,
    STAGING_PATH
)
from ..utils import add_prefix_to_caption

# Import these conditionally to avoid import errors if not available
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
except ImportError:
    # Set dummy values so we can still run with mock model
    IMAGE_TOKEN_INDEX = DEFAULT_IMAGE_TOKEN = DEFAULT_IM_START_TOKEN = DEFAULT_IM_END_TOKEN = ""

logger = logging.getLogger(__name__)

@dataclass
class CaptioningProgress:
    """Track progress of captioning for a sample"""
    sample_key: str
    media_type: str  # "video" or "image"
    total_frames: int
    processed_frames: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class WebDatasetCaptioningService:
    """Service for captioning media in WebDataset shards"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _image_processor = None
    _model_loading = None
    _loop = None
    
    def __new__(cls, model_name=CAPTIONING_MODEL):
        if cls._instance is not None:
            return cls._instance
        
        instance = super().__new__(cls)
        if PRELOAD_CAPTIONING_MODEL:
            cls._instance = instance
            try:
                cls._loop = asyncio.get_running_loop()
            except RuntimeError:
                cls._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(cls._loop)
            
            if not USE_MOCK_CAPTIONING_MODEL and cls._model_loading is None:
                cls._model_loading = cls._loop.create_task(cls._background_load_model(model_name))
        return instance
    
    def __init__(self, model_name=CAPTIONING_MODEL):
        if hasattr(self, 'model_name'):  # Already initialized
            return
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.active_tasks = {}
        self._should_stop = False
        self._model_loaded = False
        self.wds_manager = WebDatasetManager(STORAGE_PATH)
    
    @classmethod
    async def _background_load_model(cls, model_name):
        """Background task to load the model"""
        try:
            logger.info("Starting background model loading...")
            if not cls._loop:
                cls._loop = asyncio.get_running_loop()
            
            def load_model():
                try:
                    tokenizer, model, image_processor, _ = load_pretrained_model(
                        model_name, None, "llava_qwen", 
                        torch_dtype="bfloat16", device_map="auto"
                    )
                    model.eval()
                    return tokenizer, model, image_processor
                except Exception as e:
                    logger.error(f"Error in load_model: {str(e)}")
                    raise

            result = await cls._loop.run_in_executor(None, load_model)
            
            cls._tokenizer, cls._model, cls._image_processor = result
            logger.info("Background model loading completed successfully!")
            
        except Exception as e:
            logger.error(f"Background model loading failed: {str(e)}")
            cls._model_loading = None
            raise
    
    async def ensure_model_loaded(self):
        """Ensure model is loaded before processing"""
        if USE_MOCK_CAPTIONING_MODEL:
            logger.info("Using mock model, skipping model loading")
            self.__class__._model_loading = None
            self._model_loaded = True
            return

        if not self._model_loaded:
            try:
                if PRELOAD_CAPTIONING_MODEL and self.__class__._model_loading:
                    logger.info("Waiting for background model loading to complete...")
                    if self.__class__._loop and self.__class__._loop != asyncio.get_running_loop():
                        logger.warning("Different event loop detected, creating new loading task")
                        self.__class__._model_loading = None
                        await self._load_model_sync()
                    else:
                        await self.__class__._model_loading
                        self.model = self.__class__._model
                        self.tokenizer = self.__class__._tokenizer
                        self.image_processor = self.__class__._image_processor
                else:
                    await self._load_model_sync()
                
                self._model_loaded = True
                logger.info("Model loading completed!")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
    
    async def _load_model_sync(self):
        """Synchronously load the model"""
        logger.info("Loading model synchronously...")
        current_loop = asyncio.get_running_loop()
        
        def load_model():
            return load_pretrained_model(
                self.model_name, None, "llava_qwen",
                torch_dtype="bfloat16", device_map="auto"
            )
        
        self.tokenizer, self.model, self.image_processor, _ = await current_loop.run_in_executor(
            None, load_model
        )
        self.model.eval()
    
    def _load_video(self, video_data: bytes, max_frames_num: int = 64, fps: int = 1, force_sample: bool = True) -> Tuple[np.ndarray, str, float]:
        """Load and preprocess video frames from bytes
        
        Args:
            video_data: Video data as bytes
            max_frames_num: Maximum number of frames to extract (default: 64)
            fps: Frames per second to sample (default: 1)
            force_sample: Whether to force uniform sampling (default: True)
            
        Returns:
            Tuple of (frames, frame_times_str, video_time)
        """
        # Write video to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name
        
        try:
            video_path_str = tmp_path
            logger.debug(f"Loading video from temp file: {video_path_str}")
            
            # Handle empty video case
            if max_frames_num == 0:
                return np.zeros((1, 336, 336, 3)), "", 0
                
            vr = VideoReader(video_path_str, ctx=cpu(0), num_threads=1)
            total_frame_num = len(vr)
            video_time = total_frame_num / vr.get_avg_fps()
            
            # Calculate frame indices with uniform sampling
            fps = round(vr.get_avg_fps()/fps)
            frame_idx = [i for i in range(0, len(vr), fps)]
            frame_time = [i/fps for i in frame_idx]
            
            # Force uniform sampling if too many frames
            if len(frame_idx) > max_frames_num or force_sample:
                sample_fps = max_frames_num
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i/vr.get_avg_fps() for i in frame_idx]
            
            frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
            
            try:
                frames = vr.get_batch(frame_idx).asnumpy()
                logger.debug(f"Loaded {len(frames)} frames with shape {frames.shape}")
                return frames, frame_time_str, video_time
            except Exception as e:
                logger.error(f"Error loading video frames: {str(e)}")
                raise
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    async def update_sample_caption(self, shard_path: Path, sample_key: str, caption: str) -> bool:
        """Update caption for a sample in a shard
        
        Args:
            shard_path: Path to shard containing the sample
            sample_key: Key of the sample to update
            caption: New caption text
            
        Returns:
            bool: Success status
        """
        try:
            # Open the original shard
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Create a new shard for the updated content
                new_shard_path = temp_dir / f"updated_{uuid.uuid4().hex[:8]}.tar"
                
                # Create dataset for this shard
                url = f"file:{shard_path}"
                dataset = wds.WebDataset(url)
                
                # Create writer for new shard
                writer = ShardWriter(temp_dir, f"updated_{uuid.uuid4().hex[:8]}")
                
                try:
                    # Copy all samples, updating the caption for the target sample
                    for sample in dataset:
                        if sample.get("__key__") == sample_key:
                            # Update this sample with new caption
                            updated_sample = sample.copy()
                            
                            # Add or replace caption
                            updated_sample["txt"] = caption.encode('utf-8')
                            
                            # Write the updated sample
                            for key, value in updated_sample.items():
                                if key == "__key__":
                                    continue
                                    
                                # Determine type and extension
                                if key == "txt":
                                    writer.add_sample(sample_key, "caption", "txt", value)
                                elif key in ["mp4", "webm"]:
                                    writer.add_sample(sample_key, "video", key, value)
                                elif key in ["jpg", "jpeg", "png", "webp"]:
                                    writer.add_sample(sample_key, "image", key, value)
                                else:
                                    # Other fields
                                    writer.add_sample(sample_key, "data", key, value)
                        else:
                            # Just copy this sample
                            for key, value in sample.items():
                                if key == "__key__":
                                    continue
                                    
                                # Add each field to the new shard
                                if key == "txt":
                                    writer.add_sample(sample["__key__"], "caption", "txt", value)
                                elif key in ["mp4", "webm"]:
                                    writer.add_sample(sample["__key__"], "video", key, value)
                                elif key in ["jpg", "jpeg", "png", "webp"]:
                                    writer.add_sample(sample["__key__"], "image", key, value)
                                else:
                                    # Other fields
                                    writer.add_sample(sample["__key__"], "data", key, value)
                finally:
                    writer.close()
                
                # Replace the original shard with the new one
                shutil.copy2(new_shard_path, shard_path)
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating caption in {shard_path}: {e}")
            return False
    
    def update_file_caption(self, file_path: Path, caption: str) -> bool:
        """Update caption for a file in the staging directory
        
        Args:
            file_path: Path to the media file
            caption: New caption text
            
        Returns:
            bool: Success status
        """
        try:
            # Create the caption file path
            caption_path = file_path.with_suffix('.txt')
            
            # Write the new caption
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # Update in WebDataset shards if possible
            key = file_path.stem
            processed_shards = self.wds_manager.list_shards("processed")
            
            for shard_path in processed_shards:
                metadata = self.wds_manager.get_shard_metadata(shard_path)
                samples = metadata.get("samples", {})
                
                # Look for sample with matching key or similar key
                matching_key = None
                for sample_key in samples.keys():
                    if sample_key == key or sample_key.startswith(f"{key}_"):
                        matching_key = sample_key
                        break
                
                if matching_key:
                    self.update_sample_caption(shard_path, matching_key, caption)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating caption for {file_path}: {e}")
            return False
    
    def list_samples_to_caption(self) -> List[List[str]]:
        """List all samples that need captioning
        
        Returns:
            List of [sample_key, status] pairs
        """
        results = []
        
        # List all processed shards (after scene detection)
        processed_shards = self.wds_manager.list_shards("processed")
        
        for shard_path in processed_shards:
            # Extract metadata to find samples
            metadata = self.wds_manager.get_shard_metadata(shard_path)
            
            if not metadata or "samples" not in metadata:
                continue
            
            # Find samples
            for key, info in metadata.get("samples", {}).items():
                # Check if this is a media sample
                is_video = info.get("type") == "video"
                is_image = info.get("type") == "image"
                
                if not (is_video or is_image):
                    continue
                
                # Check if it has a caption
                has_caption = info.get("has_caption", False)
                
                sample_type = "video" if is_video else "image"
                status = "captioned" if has_caption else f"no caption ({sample_type})"
                
                # Add to results list
                results.append([key, status])
        
        # Also check files in the staging directory for compatibility with existing system
        for file_path in STAGING_PATH.glob("*.*"):
            if is_video_file(file_path) or is_image_file(file_path):
                caption_path = file_path.with_suffix('.txt')
                has_caption = caption_path.exists() and caption_path.stat().st_size > 0
                
                media_type = "video" if is_video_file(file_path) else "image"
                status = "captioned" if has_caption else f"no caption ({media_type})"
                
                results.append([file_path.name, status])
        
        # Sort by key and remove duplicates
        unique_results = {}
        for key, status in results:
            # Prefer captioned status over non-captioned
            if key in unique_results and "captioned" in unique_results[key]:
                continue
            unique_results[key] = status
        
        # Sort by key
        return sorted([[key, status] for key, status in unique_results.items()], key=lambda x: x[0])
    
    async def process_video_sample(self, sample: Dict[str, Any], prompt: str, prompt_prefix: str = "") -> AsyncGenerator[Tuple[CaptioningProgress, Optional[str]], None]:
        """Process a video sample for captioning
        
        Args:
            sample: WebDataset sample containing video
            prompt: Prompt for captioning model
            prompt_prefix: Optional prefix to add to all captions
            
        Yields:
            Tuples of (progress, caption)
        """
        key = sample.get("__key__", "unknown")
        
        # Find video content
        video_data = None
        video_ext = None
        
        for ext in ['.mp4', '.webm']:
            ext_key = ext[1:]  # Remove the dot
            if ext_key in sample:
                video_data = sample[ext_key]
                video_ext = ext_key
                break
        
        if not video_data:
            progress = CaptioningProgress(
                sample_key=key,
                media_type="video",
                total_frames=0,
                processed_frames=0,
                status="error",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error="No video content found"
            )
            yield progress, None
            return
        
        # Initialize progress
        progress = CaptioningProgress(
            sample_key=key,
            media_type="video",
            total_frames=0,
            processed_frames=0,
            status="initializing",
            started_at=datetime.now()
        )
        self.active_tasks[key] = progress
        yield progress, None
        
        # Get any existing caption
        existing_caption = None
        for ext in ['.txt', '.caption', '.json', '.cls']:
            ext_key = ext[1:]  # Remove the dot
            if ext_key in sample:
                if isinstance(sample[ext_key], bytes):
                    existing_caption = sample[ext_key].decode('utf-8')
                else:
                    existing_caption = sample[ext_key]
                break
        
        # Ensure model is loaded before processing
        await self.ensure_model_loaded()
        
        if USE_MOCK_CAPTIONING_MODEL:
            # Mock mode - generate a simple caption
            clip_caption = f"This is a test caption for video {key}"
            
            if prompt_prefix and not clip_caption.startswith(prompt_prefix):
                clip_caption = f"{prompt_prefix}{clip_caption}"
            
            progress.status = "completed"
            progress.processed_frames = 1
            progress.total_frames = 1
            progress.completed_at = datetime.now()
            yield progress, clip_caption
            return
        
        try:
            # Process video frames
            loop = asyncio.get_event_loop()
            frames, frame_times_str, video_time = await loop.run_in_executor(
                None, 
                lambda: self._load_video(video_data, max_frames_num=64, fps=1, force_sample=True)
            )
            
            # Update progress
            progress.total_frames = len(frames)
            progress.processed_frames = 0
            progress.status = "processing frames"
            yield progress, None
            
            # Process frames with the image processor
            processed_frames = await loop.run_in_executor(
                None,
                lambda: self.image_processor.preprocess(
                    frames, 
                    return_tensors="pt"
                )["pixel_values"]
            )
            
            # Update progress
            progress.processed_frames = len(frames)
            progress.status = "generating caption"
            yield progress, None
            
            # Move to GPU
            video_tensor = processed_frames.to('cuda').bfloat16()
            
            # Use proper conversation template and tokens
            conv_template = "qwen_1_5"
            time_instruction = (f"The video lasts for {video_time:.2f} seconds, and {len(frames)} "
                            f"frames are uniformly sampled from it. These frames are located at {frame_times_str}.")
            
            full_question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n{prompt}"
            
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], full_question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            # Cap the output length to prevent hallucination
            max_new_tokens = 512  # Reasonable limit for caption length
            
            input_ids = await loop.run_in_executor(
                None,
                lambda: tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
            )
            
            # Generate caption with controlled parameters
            with torch.no_grad():
                output = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_ids,
                        images=[video_tensor],
                        modalities=["video"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=max_new_tokens,
                    )
                )
                
                clip_caption = await loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
                )
                
                # Remove the instruction/question part from the response
                if time_instruction in clip_caption:
                    clip_caption = clip_caption.split(time_instruction)[1].strip()
                if prompt in clip_caption:
                    clip_caption = clip_caption.split(prompt)[1].strip()
            
            # Add prefix if needed
            if prompt_prefix and not clip_caption.startswith(prompt_prefix):
                clip_caption = f"{prompt_prefix}{clip_caption}"
            
            progress.status = "completed"
            progress.completed_at = datetime.now()
            yield progress, clip_caption
            
        except Exception as e:
            logger.error(f"Error processing video sample {key}: {e}")
            progress.status = "error"
            progress.error = str(e)
            progress.completed_at = datetime.now()
            yield progress, None
    
    async def process_image_sample(self, sample: Dict[str, Any], prompt: str, prompt_prefix: str = "") -> AsyncGenerator[Tuple[CaptioningProgress, Optional[str]], None]:
        """Process an image sample for captioning
        
        Args:
            sample: WebDataset sample containing image
            prompt: Prompt for captioning model
            prompt_prefix: Optional prefix to add to all captions
            
        Yields:
            Tuples of (progress, caption)
        """
        key = sample.get("__key__", "unknown")
        
        # Find image content
        image_data = None
        image_ext = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            ext_key = ext[1:]  # Remove the dot
            if ext_key in sample:
                image_data = sample[ext_key]
                image_ext = ext_key
                break
        
        if not image_data:
            progress = CaptioningProgress(
                sample_key=key,
                media_type="image",
                total_frames=1,
                processed_frames=0,
                status="error",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error="No image content found"
            )
            yield progress, None
            return
        
        # Initialize progress
        progress = CaptioningProgress(
            sample_key=key,
            media_type="image",
            total_frames=1,
            processed_frames=0,
            status="initializing",
            started_at=datetime.now()
        )
        self.active_tasks[key] = progress
        yield progress, None
        
        # Get any existing caption
        existing_caption = None
        for ext in ['.txt', '.caption', '.json', '.cls']:
            ext_key = ext[1:]  # Remove the dot
            if ext_key in sample:
                if isinstance(sample[ext_key], bytes):
                    existing_caption = sample[ext_key].decode('utf-8')
                else:
                    existing_caption = sample[ext_key]
                break
        
        # Ensure model is loaded before processing
        await self.ensure_model_loaded()
        
        if USE_MOCK_CAPTIONING_MODEL:
            # Mock mode - generate a simple caption
            image_caption = f"This is a test caption for image {key}"
            
            if prompt_prefix and not image_caption.startswith(prompt_prefix):
                image_caption = f"{prompt_prefix}{image_caption}"
            
            progress.status = "completed"
            progress.processed_frames = 1
            progress.total_frames = 1
            progress.completed_at = datetime.now()
            yield progress, image_caption
            return
        
        try:
            # Convert image bytes to PIL.Image
            loop = asyncio.get_event_loop()
            pil_image = await loop.run_in_executor(
                None,
                lambda: Image.open(io.BytesIO(image_data)).convert("RGB")
            )
            
            # Update progress
            progress.processed_frames = 1
            progress.status = "processing image"
            yield progress, None
            
            # Process image with the image processor
            processed_image = await loop.run_in_executor(
                None,
                lambda: self.image_processor.preprocess(
                    pil_image,
                    return_tensors="pt"
                )["pixel_values"]
            )
            
            # Update progress
            progress.status = "generating caption"
            yield progress, None
            
            # Move to GPU
            image_tensor = processed_image.to('cuda').bfloat16()
            
            # Prepare prompt
            full_prompt = f"<image>{prompt}"
            
            input_ids = await loop.run_in_executor(
                None,
                lambda: tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
            )
            
            # Generate caption
            with torch.no_grad():
                output = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_ids,
                        images=[image_tensor],
                        modalities=["image"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=512,
                    )
                )
                
                image_caption = await loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
                )
                
                # Remove the prompt part from the response
                if prompt in image_caption:
                    image_caption = image_caption.split(prompt)[1].strip()
            
            # Add prefix if needed
            if prompt_prefix and not image_caption.startswith(prompt_prefix):
                image_caption = f"{prompt_prefix}{image_caption}"
            
            progress.status = "completed"
            progress.completed_at = datetime.now()
            yield progress, image_caption
            
        except Exception as e:
            logger.error(f"Error processing image sample {key}: {e}")
            progress.status = "error"
            progress.error = str(e)
            progress.completed_at = datetime.now()
            yield progress, None
    
    def stop_captioning(self):
        """Stop all ongoing captioning tasks"""
        logger.info("Stopping all captioning tasks")
        self._should_stop = True

    async def start_caption_generation(self, custom_prompt: str, prompt_prefix: str) -> AsyncGenerator[List[List[str]], None]:
        """Generate captions for all uncaptioned samples
        
        Args:
            custom_prompt: Custom prompt to use for captioning
            prompt_prefix: Prefix to add to all captions
            
        Yields:
            Lists of [sample_key, status] pairs for UI updates
        """
        try:
            logger.info("Starting auto-caption generation")
            
            # Use provided prompt or default
            default_prompt = DEFAULT_CAPTIONING_BOT_INSTRUCTIONS
            prompt = custom_prompt.strip() or default_prompt
            logger.debug(f"Using prompt: {prompt}")
            
            # Find samples needing captions
            samples_to_process = []
            
            # List all processed shards
            processed_shards = self.wds_manager.list_shards("processed")
            
            # First pass: find all samples without captions from WebDataset shards
            for shard_path in processed_shards:
                # Create dataset for this shard
                url = f"file:{shard_path}"
                dataset = wds.WebDataset(url)
                
                for sample in dataset:
                    key = sample.get("__key__", "")
                    if not key:
                        continue
                    
                    # Check if this sample has media
                    has_image = any(ext[1:] in sample for ext in ['.jpg', '.jpeg', '.png', '.webp'])
                    has_video = any(ext[1:] in sample for ext in ['.mp4', '.webm'])
                    
                    if not (has_image or has_video):
                        continue
                    
                    # Check if this sample has a caption
                    has_caption = any(ext[1:] in sample for ext in ['.txt', '.caption', '.json', '.cls'])
                    
                    if not has_caption:
                        samples_to_process.append((shard_path, sample))
            
            # Also check files in the staging directory for compatibility with existing system
            staging_files = []
            for file_path in STAGING_PATH.glob("*.*"):
                if is_video_file(file_path) or is_image_file(file_path):
                    caption_path = file_path.with_suffix('.txt')
                    if not (caption_path.exists() and caption_path.stat().st_size > 0):
                        staging_files.append(file_path)
            
            if not samples_to_process and not staging_files:
                logger.info("No samples need captioning")
                yield []
                return
            
            # Reset stop flag
            self._should_stop = False
            self.active_tasks.clear()
            status_update = {}
            
            # Create a new training shard writer
            training_dir = self.wds_manager.get_shard_directory("training")
            training_writer = ShardWriter(training_dir, f"training_{uuid.uuid4().hex[:8]}")
            
            try:
                # Process each WebDataset sample
                for shard_path, sample in samples_to_process:
                    if self._should_stop:
                        break
                    
                    key = sample.get("__key__", "")
                    
                    try:
                        # Determine if this is image or video
                        is_video = any(ext[1:] in sample for ext in ['.mp4', '.webm'])
                        is_image = any(ext[1:] in sample for ext in ['.jpg', '.jpeg', '.png', '.webp'])
                        
                        # Process based on media type
                        if is_video:
                            process_gen = self.process_video_sample(sample, prompt, prompt_prefix)
                        elif is_image:
                            process_gen = self.process_image_sample(sample, prompt, prompt_prefix)
                        else:
                            continue
                        
                        # Process the sample and get caption
                        caption = None
                        async for progress, new_caption in process_gen:
                            if new_caption:
                                caption = new_caption
                            
                            # Store progress info
                            status_update[key] = {
                                "status": progress.status,
                                "frames": progress.processed_frames,
                                "total": progress.total_frames,
                                "type": progress.media_type
                            }
                            
                            # Convert to list format for Gradio DataFrame
                            rows = []
                            for sample_key, info in status_update.items():
                                status = info["status"]
                                if status == "processing frames":
                                    percent = (info["frames"] / info["total"]) * 100
                                    status = f"Analyzing... {percent:.1f}% ({info['frames']}/{info['total']} frames)"
                                elif status == "generating caption":
                                    status = "Generating caption..."
                                elif status == "error":
                                    status = f"Error: {progress.error}"
                                elif status == "completed":
                                    status = f"Completed ({info['type']})"
                                
                                rows.append([sample_key, status])
                            
                            yield rows
                            await asyncio.sleep(0.1)
                        
                        # If caption was generated, add the updated sample to the training shard
                        if caption:
                            # Copy all fields from the original sample
                            for field_key, value in sample.items():
                                if field_key == "__key__":
                                    continue
                                
                                # Determine field type
                                if field_key in ["mp4", "webm"]:
                                    training_writer.add_sample(key, "video", field_key, value)
                                elif field_key in ["jpg", "jpeg", "png", "webp"]:
                                    training_writer.add_sample(key, "image", field_key, value)
                                elif field_key not in ["txt", "caption", "json", "cls"]:
                                    training_writer.add_sample(key, "data", field_key, value)
                            
                            # Add the new caption
                            training_writer.add_sample(key, "caption", "txt", caption.encode('utf-8'))
                            
                            # Update original shard too
                            await self.update_sample_caption(shard_path, key, caption)
                    
                    except Exception as e:
                        logger.error(f"Error processing sample {key}: {e}")
                        status_update[key] = {
                            "status": "error",
                            "frames": 0,
                            "total": 1,
                            "type": "unknown"
                        }
                        rows = [[str(key), f"Error: {str(e)}"]]
                        yield rows
                        continue
                
                # Process files in staging directory
                for file_path in staging_files:
                    if self._should_stop:
                        break
                    
                    key = file_path.name
                    
                    try:
                        if is_video_file(file_path):
                            # Create a sample for the video
                            sample = {"__key__": file_path.stem}
                            with open(file_path, 'rb') as f:
                                video_data = f.read()
                            sample[file_path.suffix.lstrip('.')] = video_data
                            
                            # Caption the video
                            process_gen = self.process_video_sample(sample, prompt, prompt_prefix)
                        elif is_image_file(file_path):
                            # Create a sample for the image
                            sample = {"__key__": file_path.stem}
                            with open(file_path, 'rb') as f:
                                image_data = f.read()
                            sample[file_path.suffix.lstrip('.')] = image_data
                            
                            # Caption the image
                            process_gen = self.process_image_sample(sample, prompt, prompt_prefix)
                        else:
                            continue
                        
                        # Process the sample and get caption
                        caption = None
                        async for progress, new_caption in process_gen:
                            if new_caption:
                                caption = new_caption
                            
                            # Store progress info
                            status_update[key] = {
                                "status": progress.status,
                                "frames": progress.processed_frames,
                                "total": progress.total_frames,
                                "type": progress.media_type
                            }
                            
                            # Convert to list format for Gradio DataFrame
                            rows = []
                            for sample_key, info in status_update.items():
                                status = info["status"]
                                if status == "processing frames":
                                    percent = (info["frames"] / info["total"]) * 100
                                    status = f"Analyzing... {percent:.1f}% ({info['frames']}/{info['total']} frames)"
                                elif status == "generating caption":
                                    status = "Generating caption..."
                                elif status == "error":
                                    status = f"Error: {progress.error}"
                                elif status == "completed":
                                    status = f"Completed ({info['type']})"
                                
                                rows.append([sample_key, status])
                            
                            yield rows
                            await asyncio.sleep(0.1)
                        
                        # If caption was generated, write it to file
                        if caption:
                            caption_path = file_path.with_suffix('.txt')
                            with open(caption_path, 'w', encoding='utf-8') as f:
                                f.write(caption)
                            
                            # Also add to training shard
                            sample_key = file_path.stem
                            media_ext = file_path.suffix.lstrip('.')
                            
                            if is_video_file(file_path):
                                training_writer.add_sample(sample_key, "video", media_ext, video_data)
                            elif is_image_file(file_path):
                                training_writer.add_sample(sample_key, "image", media_ext, image_data)
                            
                            training_writer.add_sample(sample_key, "caption", "txt", caption.encode('utf-8'))
                            
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        status_update[key] = {
                            "status": "error",
                            "frames": 0,
                            "total": 1,
                            "type": "unknown"
                        }
                        rows = [[str(key), f"Error: {str(e)}"]]
                        yield rows
                        continue
            
            finally:
                # Close the training shard writer
                training_writer.close()
            
            logger.info("Caption generation completed")
            
            # Final update with fresh data
            final_results = self.list_samples_to_caption()
            yield final_results
            
        except Exception as e:
            logger.error(f"Error in caption generation: {str(e)}")
            rows = [[f"Error: {str(e)}", "error"]]
            yield rows
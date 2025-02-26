import logging
import torch
import shutil
import gradio as gr
import numpy as np
from decord import VideoReader, cpu
from pathlib import Path
from typing import Any, Tuple, Dict, Optional, AsyncGenerator, List
import asyncio
from dataclasses import dataclass
from datetime import datetime
import cv2
import copy

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from .config import TRAINING_VIDEOS_PATH, STAGING_PATH, PRELOAD_CAPTIONING_MODEL, CAPTIONING_MODEL, USE_MOCK_CAPTIONING_MODEL, DEFAULT_CAPTIONING_BOT_INSTRUCTIONS, VIDEOS_TO_SPLIT_PATH, DEFAULT_PROMPT_PREFIX
from .utils import extract_scene_info, is_image_file, is_video_file
from .finetrainers_utils import copy_files_to_training_dir, prepare_finetrainers_dataset

logger = logging.getLogger(__name__)

@dataclass
class CaptioningProgress:
    video_name: str
    total_frames: int
    processed_frames: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class CaptioningService:
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
        self.active_tasks: Dict[str, CaptioningProgress] = {}
        self._should_stop = False
        self._model_loaded = False

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


    def update_file_caption(self, file_path: Path, caption: str) -> None:
        """Update caption for a training file
        
        Args:
            file_path: Path to the media file
            caption: New caption text
        """
        try:
            # Ensure we're working with Path objects
            file_path = Path(file_path)
            
            # Create the caption file path
            caption_path = file_path.with_suffix('.txt')
            
            # Write the new caption
            caption_path.write_text(caption)
            
            logger.info(f"Updated caption for {file_path.name}")
            
            # the following code is disabled, because we want to make the copy to prompts.txt manual
            # If the file is in TRAINING_VIDEOS_PATH, update prompts.txt as well
            # if TRAINING_VIDEOS_PATH in file_path.parents:
            #     # Try to update the training dataset
            #     try:
            #         prepare_finetrainers_dataset()
            #         logger.info("Updated training dataset with new caption")
            #     except Exception as e:
            #         logger.warning(f"Could not update training dataset: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error updating caption: {str(e)}")
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
    
    def _load_video(self, video_path: Path, max_frames_num: int = 64, fps: int = 1, force_sample: bool = True) -> tuple[np.ndarray, str, float]:
        """Load and preprocess video frames with strict limits
        
        Args:
            video_path: Path to video file
            max_frames_num: Maximum number of frames to extract (default: 64)
            fps: Frames per second to sample (default: 1)
            force_sample: Whether to force uniform sampling (default: True)
            
        Returns:
            Tuple of (frames, frame_times_str, video_time)
        """
        video_path_str = str(video_path)
        logger.debug(f"Loading video: {video_path_str}")
        
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

    async def process_video(self, video_path: Path, prompt: str, prompt_prefix: str = "") -> AsyncGenerator[tuple[CaptioningProgress, Optional[str]], None]:
        try:
            video_name = video_path.name
            logger.info(f"Starting processing of video: {video_name}")
            
            # Load video metadata with strict frame limits
            logger.debug(f"Loading video metadata for {video_name}")
            loop = asyncio.get_event_loop()
            vr = await loop.run_in_executor(None, lambda: VideoReader(str(video_path), ctx=cpu(0)))
            total_frames = len(vr)
            
            progress = CaptioningProgress(
                video_name=video_name,
                total_frames=total_frames,
                processed_frames=0,
                status="initializing",
                started_at=datetime.now()
            )
            self.active_tasks[video_name] = progress
            yield progress, None

            # Get parent caption if this is a clip
            parent_caption = ""
            if "___" in video_path.stem:
                parent_name, _ = extract_scene_info(video_path.stem)
                parent_txt_path = VIDEOS_TO_SPLIT_PATH / f"{parent_name}.txt"
                if parent_txt_path.exists():
                    parent_caption = parent_txt_path.read_text().strip()

            # Ensure model is loaded before processing
            await self.ensure_model_loaded()

            if USE_MOCK_CAPTIONING_MODEL:
                # Even in mock mode, we'll generate a caption that shows we processed parent info
                clip_caption = f"This is a test caption for {video_name}"

                # Combine clip caption with parent caption
                if parent_caption:
                    full_caption = f"{clip_caption}\n{parent_caption}"
                else:
                    full_caption = clip_caption

                if prompt_prefix and not full_caption.startswith(prompt_prefix):
                    full_caption = f"{prompt_prefix}{full_caption}"
                    
                # Write the caption file
                txt_path = video_path.with_suffix('.txt')
                txt_path.write_text(full_caption)
                
                logger.debug(f"Mock mode: Saved caption to {txt_path}")

                progress.status = "completed"
                progress.processed_frames = total_frames
                progress.completed_at = datetime.now()
                yield progress, full_caption
            else:
                # Process frames with strict limits
                max_frames_num = 64  # Maximum frames supported by the model
                frames, frame_times_str, video_time = await loop.run_in_executor(
                    None, 
                    lambda: self._load_video(video_path, max_frames_num, fps=1, force_sample=True)
                )
                
                # Process all frames at once using the image processor
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

                # Move processed frames to GPU
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

                # Combine captions with proper formatting
                if parent_caption:
                    full_caption = f"{clip_caption}\n{parent_caption}"
                else:
                    full_caption = clip_caption

                if prompt_prefix and not full_caption.startswith(prompt_prefix):
                    full_caption = f"{prompt_prefix}{full_caption}"

                # Write caption
                txt_path = video_path.with_suffix('.txt')
                txt_path.write_text(full_caption)
                
                progress.status = "completed"
                progress.completed_at = datetime.now()
                yield progress, full_caption

        except Exception as e:
            progress.status = "error"
            progress.error = str(e)
            progress.completed_at = datetime.now()
            yield progress, None
            raise

    async def process_image(self, image_path: Path, prompt: str, prompt_prefix: str = "") -> AsyncGenerator[tuple[CaptioningProgress, Optional[str]], None]:
        """Process a single image for captioning"""
        try:
            image_name = image_path.name
            logger.info(f"Starting processing of image: {image_name}")
            
            progress = CaptioningProgress(
                video_name=image_name,  # Reusing video_name field for images
                total_frames=1,
                processed_frames=0,
                status="initializing",
                started_at=datetime.now()
            )
            self.active_tasks[image_name] = progress
            yield progress, None

            # Ensure model is loaded
            await self.ensure_model_loaded()

            if USE_MOCK_CAPTIONING_MODEL:
                progress.status = "completed"
                progress.processed_frames = 1
                progress.completed_at = datetime.now()
                print("yielding fake")
                yield progress, "This is a test image caption"
                return

            # Read and process image
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                lambda: cv2.imread(str(image_path))
            )
            if image is None:
                raise ValueError(f"Could not read image: {str(image_path)}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            processed_image = await loop.run_in_executor(
                None,
                lambda: self.image_processor.preprocess(
                    image,
                    return_tensors="pt"
                )["pixel_values"]
            )
            
            progress.processed_frames = 1
            progress.status = "generating caption"
            yield progress, None

            # Move to GPU and generate caption
            image_tensor = processed_image.to('cuda').bfloat16()
            full_prompt = f"<image>{prompt}"

            input_ids = await loop.run_in_executor(
                None,
                lambda: tokenizer_image_token(full_prompt, self.tokenizer, return_tensors="pt").unsqueeze(0).to('cuda')
            )
            
            with torch.no_grad():
                output = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_ids,
                        images=[image_tensor],
                        modalities=["image"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=4096,
                    )
                )

            caption = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
            )
            
            progress.status = "completed"
            progress.completed_at = datetime.now()
            gr.Info(f"Successfully generated caption for {image_name}")
            yield progress, caption

        except Exception as e:
            progress.status = "error"
            progress.error = str(e)
            progress.completed_at = datetime.now()
            yield progress, None
            raise gr.Error(f"Error processing image: {str(e)}")

   
    async def start_caption_generation(self, custom_prompt: str, prompt_prefix: str) -> AsyncGenerator[List[List[str]], None]:
        """Iterates over clips to auto-generate captions asynchronously."""
        try:
            logger.info("Starting auto-caption generation")
        
            # Use provided prompt or default
            default_prompt = DEFAULT_CAPTIONING_BOT_INSTRUCTIONS
            prompt = custom_prompt.strip() or default_prompt
            logger.debug(f"Using prompt: {prompt}")

            # Find files needing captions
            video_files = list(STAGING_PATH.glob("*.mp4"))
            image_files = [f for f in STAGING_PATH.glob("*") if is_image_file(f)]
            all_files = video_files + image_files
            
            # Filter for files missing captions or with empty caption files
            files_to_process = []
            for file_path in all_files:
                caption_path = file_path.with_suffix('.txt')
                needs_caption = (
                    not caption_path.exists() or 
                    caption_path.stat().st_size == 0 or
                    caption_path.read_text().strip() == ""
                )
                if needs_caption:
                    files_to_process.append(file_path)
            
            logger.info(f"Found {len(files_to_process)} files needing captions")
            
            if not files_to_process:
                logger.info("No files need captioning")
                yield []
                return

            self._should_stop = False
            self.active_tasks.clear()
            status_update: Dict[str, Dict[str, Any]] = {}

            for file_path in all_files:
                if self._should_stop:
                    break

                try:
                    print(f"we are in file_path {str(file_path)}")
                    # Choose appropriate processing method based on file type
                    if is_video_file(file_path):
                        process_gen = self.process_video(file_path, prompt, prompt_prefix)
                    else:
                        process_gen = self.process_image(file_path, prompt, prompt_prefix)
                    print("got process_gen = ", process_gen)
                    async for progress, caption in process_gen:
                        print(f"process_gen contains this caption = {caption}")
                        if caption and prompt_prefix and not caption.startswith(prompt_prefix):
                            caption = f"{prompt_prefix}{caption}"
                            
                        # Save caption
                        if caption:
                            txt_path = file_path.with_suffix('.txt')
                            txt_path.write_text(caption)
                            
                        logger.debug(f"Progress update: {progress.status}")
                        
                        # Store progress info
                        status_update[file_path.name] = {
                            "status": progress.status,
                            "frames": progress.processed_frames,
                            "total": progress.total_frames
                        }

                        # Convert to list format for Gradio DataFrame
                        rows = []
                        for file_name, info in status_update.items():
                            status = info["status"]
                            if status == "processing":
                                percent = (info["frames"] / info["total"]) * 100
                                status = f"Analyzing... {percent:.1f}% ({info['frames']}/{info['total']} frames)"
                            elif status == "generating caption":
                                status = "Generating caption..."
                            elif status == "error":
                                status = f"Error: {progress.error}"
                            elif status == "completed":
                                status = "Completed"
                                
                            rows.append([file_name, status])

                        yield rows
                        await asyncio.sleep(0.1)
                    

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
                    rows = [[str(file_path.name), f"Error: {str(e)}"]]
                    yield rows
                    continue

            logger.info("Auto-caption generation completed, cyping assets to the training dir..")

            copy_files_to_training_dir(prompt_prefix)
        except Exception as e:
            logger.error(f"Error in start_caption_generation: {str(e)}")
            yield [[str(e), "error"]]
            raise

    def stop_captioning(self):
        """Stop all ongoing captioning tasks"""
        logger.info("Stopping all captioning tasks")
        self._should_stop = True

    def close(self):
        """Clean up resources"""
        logger.info("Cleaning up captioning service resources")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'image_processor'):
            del self.image_processor
        torch.cuda.empty_cache()
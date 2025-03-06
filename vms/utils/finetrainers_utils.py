import gradio as gr
from pathlib import Path
import logging
import shutil
from typing import Any, Optional, Dict, List, Union, Tuple

from ..config import (
    STORAGE_PATH, TRAINING_PATH, STAGING_PATH, TRAINING_VIDEOS_PATH, MODEL_PATH, OUTPUT_PATH, HF_API_TOKEN, MODEL_TYPES,
    DEFAULT_VALIDATION_NB_STEPS,
    DEFAULT_VALIDATION_HEIGHT,
    DEFAULT_VALIDATION_WIDTH,
    DEFAULT_VALIDATION_NB_FRAMES,
    DEFAULT_VALIDATION_FRAMERATE
)
from .utils import get_video_fps, extract_scene_info, make_archive, is_image_file, is_video_file

logger = logging.getLogger(__name__)

def prepare_finetrainers_dataset() -> Tuple[Path, Path]:
    """Prepare a Finetrainers-compatible dataset structure
    
    Creates:
        training/
        ├── prompt.txt       # All captions, one per line
        ├── videos.txt       # All video paths, one per line
        └── videos/          # Directory containing all mp4 files
            ├── 00000.mp4
            ├── 00001.mp4
            └── ...
    Returns:
        Tuple of (videos_file_path, prompts_file_path)
    """

    # Verifies the videos subdirectory
    TRAINING_VIDEOS_PATH.mkdir(exist_ok=True)
    
    # Clear existing training lists
    for f in TRAINING_PATH.glob("*"):
        if f.is_file():
            if f.name in ["videos.txt", "prompts.txt", "prompt.txt"]:
                f.unlink()
    
    videos_file = TRAINING_PATH / "videos.txt"
    prompts_file = TRAINING_PATH / "prompts.txt"  # Finetrainers can use either prompts.txt or prompt.txt
    
    media_files = []
    captions = []
    
    # Process all video files from the videos subdirectory
    for idx, file in enumerate(sorted(TRAINING_VIDEOS_PATH.glob("*.mp4"))):
        caption_file = file.with_suffix('.txt')
        if caption_file.exists():
            # Normalize caption to single line
            caption = caption_file.read_text().strip()
            caption = ' '.join(caption.split())
            
            # Use relative path from training root
            relative_path = f"videos/{file.name}"
            media_files.append(relative_path)
            captions.append(caption)

    # Write files if we have content
    if media_files and captions:
        videos_file.write_text('\n'.join(media_files))
        prompts_file.write_text('\n'.join(captions))
        logger.info(f"Created dataset with {len(media_files)} video/caption pairs")
    else:
        logger.warning("No valid video/caption pairs found in training directory")
        return None, None
        
    # Verify file contents
    with open(videos_file) as vf:
        video_lines = [l.strip() for l in vf.readlines() if l.strip()]
    with open(prompts_file) as pf:
        prompt_lines = [l.strip() for l in pf.readlines() if l.strip()]
        
    if len(video_lines) != len(prompt_lines):
        logger.error(f"Mismatch in generated files: {len(video_lines)} videos vs {len(prompt_lines)} prompts")
        return None, None
        
    return videos_file, prompts_file

def copy_files_to_training_dir(prompt_prefix: str) -> int:
    """Just copy files over, with no destruction"""

    gr.Info("Copying assets to the training dataset..")

    # Find files needing captions
    video_files = list(STAGING_PATH.glob("*.mp4"))
    image_files = [f for f in STAGING_PATH.glob("*") if is_image_file(f)]
    all_files = video_files + image_files
    
    nb_copied_pairs = 0

    for file_path in all_files:

        caption = ""
        file_caption_path = file_path.with_suffix('.txt')
        if file_caption_path.exists():
            logger.debug(f"Found caption file: {file_caption_path}")
            caption = file_caption_path.read_text()

         # Get parent caption if this is a clip
        parent_caption = ""
        if "___" in file_path.stem:
            parent_name, _ = extract_scene_info(file_path.stem)
            #print(f"parent_name is {parent_name}")
            parent_caption_path = STAGING_PATH / f"{parent_name}.txt"
            if parent_caption_path.exists():
                logger.debug(f"Found parent caption file: {parent_caption_path}")
                parent_caption = parent_caption_path.read_text().strip()

        target_file_path = TRAINING_VIDEOS_PATH / file_path.name

        target_caption_path = target_file_path.with_suffix('.txt')

        if parent_caption and not caption.endswith(parent_caption):
            caption = f"{caption}\n{parent_caption}"

        # Add FPS information for videos
        if is_video_file(file_path) and caption:
            # Only add FPS if not already present
            if not any(f"FPS, " in line for line in caption.split('\n')):
                fps_info = get_video_fps(file_path)
                if fps_info:
                    caption = f"{fps_info}{caption}"

        if prompt_prefix and not caption.startswith(prompt_prefix):
            caption = f"{prompt_prefix}{caption}"
            
        # make sure we only copy over VALID pairs
        if caption:
            try:
                target_caption_path.write_text(caption)
                shutil.copy2(file_path, target_file_path)
                nb_copied_pairs += 1
            except Exception as e:
                print(f"failed to copy one of the pairs: {e}")
                pass

    prepare_finetrainers_dataset()

    gr.Info(f"Successfully generated the training dataset ({nb_copied_pairs} pairs)")

    return nb_copied_pairs

# Add this function to finetrainers_utils.py or a suitable place

def create_validation_config() -> Optional[Path]:
    """Create a validation configuration JSON file for Finetrainers
    
    Creates a validation dataset file with a subset of the training data
    
    Returns:
        Path to the validation JSON file, or None if no training files exist
    """
    # Ensure training dataset exists
    if not TRAINING_VIDEOS_PATH.exists() or not any(TRAINING_VIDEOS_PATH.glob("*.mp4")):
        logger.warning("No training videos found for validation")
        return None
    
    # Get a subset of the training videos (up to 4) for validation
    training_videos = list(TRAINING_VIDEOS_PATH.glob("*.mp4"))
    validation_videos = training_videos[:min(4, len(training_videos))]
    
    if not validation_videos:
        logger.warning("No validation videos selected")
        return None
    
    # Create validation data entries
    validation_data = {"data": []}
    
    for video_path in validation_videos:
        # Get caption from matching text file
        caption_path = video_path.with_suffix('.txt')
        if not caption_path.exists():
            logger.warning(f"Missing caption for {video_path}, skipping for validation")
            continue
            
        caption = caption_path.read_text().strip()
        
        # Get video dimensions and properties
        try:
            # Use the most common default resolution and settings
            data_entry = {
                "caption": caption,
                "image_path": "",  # No input image for text-to-video
                "video_path": str(video_path),
                "num_inference_steps": DEFAULT_VALIDATION_NB_STEPS,
                "height": DEFAULT_VALIDATION_HEIGHT,
                "width": DEFAULT_VALIDATION_WIDTH,
                "num_frames": DEFAULT_VALIDATION_NB_FRAMES,
                "frame_rate": DEFAULT_VALIDATION_FRAMERATE
            }
            validation_data["data"].append(data_entry)
        except Exception as e:
            logger.warning(f"Error adding validation entry for {video_path}: {e}")
    
    if not validation_data["data"]:
        logger.warning("No valid validation entries created")
        return None
    
    # Write validation config to file
    validation_file = OUTPUT_PATH / "validation_config.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    logger.info(f"Created validation config with {len(validation_data['data'])} entries")
    return validation_file

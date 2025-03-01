import gradio as gr
from pathlib import Path
import logging
import shutil
from typing import Any, Optional, Dict, List, Union, Tuple

from ..config import STORAGE_PATH, TRAINING_PATH, STAGING_PATH, TRAINING_VIDEOS_PATH, MODEL_PATH, OUTPUT_PATH, HF_API_TOKEN, MODEL_TYPES
from .utils import get_video_fps, extract_scene_info, make_archive, is_image_file, is_video_file

logger = logging.getLogger(__name__)

def prepare_finetrainers_dataset() -> Tuple[Path, Path]:
    """make sure we have a Finetrainers-compatible dataset structure
    
    Checks that we have:
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
            if f.name in ["videos.txt", "prompts.txt"]:
                f.unlink()
    
    videos_file = TRAINING_PATH / "videos.txt"
    prompts_file = TRAINING_PATH / "prompts.txt"  # Note: Changed from prompt.txt to prompts.txt to match our config
    
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
            
            # Clean up the caption file since it's now in prompts.txt
            # EDIT well you know what, let's keep it, otherwise running the function
            # twice might cause some errors
            # caption_file.unlink()

    # Write files if we have content
    if media_files and captions:
        videos_file.write_text('\n'.join(media_files))
        prompts_file.write_text('\n'.join(captions))
  
    else:
        raise ValueError("No valid video/caption pairs found in training directory")
    # Verify file contents
    with open(videos_file) as vf:
        video_lines = [l.strip() for l in vf.readlines() if l.strip()]
    with open(prompts_file) as pf:
        prompt_lines = [l.strip() for l in pf.readlines() if l.strip()]
        
    if len(video_lines) != len(prompt_lines):
        raise ValueError(f"Mismatch in generated files: {len(video_lines)} videos vs {len(prompt_lines)} prompts")
        
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

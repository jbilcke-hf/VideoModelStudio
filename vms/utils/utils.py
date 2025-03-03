import os
import shutil
import subprocess
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import json
import re
import logging
from typing import Any, Optional, Dict, List, Union, Tuple

logger = logging.getLogger(__name__)

def make_archive(source: str | Path, destination: str | Path):
    source = str(source)
    destination = str(destination)
    #print(f"make_archive({source}, {destination})")
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s'%(name,format), destination)

def get_video_fps(video_path: Path) -> Optional[str]:
    """Get FPS information from video file using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        FPS string (e.g. "24 FPS, ") or None if unable to determine
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Error getting FPS for {video_path}: {result.stderr}")
            return None
            
        fps = result.stdout.strip()
        if '/' in fps:
            # Convert fraction to decimal
            num, den = map(int, fps.split('/'))
            if den == 0:
                return None
            fps = str(round(num / den))
            
        return f"{fps} FPS, "
        
    except Exception as e:
        logger.warning(f"Failed to get FPS for {video_path}: {e}")
        return None

def extract_scene_info(filename: str) -> Tuple[str, Optional[int]]:
    """Extract base name and scene number from filename
    
    Args:
        filename: Input filename like "my_cool_video_1___001.mp4"
        
    Returns:
        Tuple of (base_name, scene_number)
        e.g. ("my_cool_video_1", 1)
    """
    # Match numbers at the end of the filename before extension
    match = re.search(r'(.+?)___(\d+)$', Path(filename).stem)
    if match:
        return match.group(1), int(match.group(2))
    return Path(filename).stem, None

def is_image_file(file_path: Path) -> bool:
    """Check if file is an image based on extension
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if file has image extension
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.heic'}
    return file_path.suffix.lower() in image_extensions

def is_video_file(file_path: Path) -> bool:
    """Check if file is a video based on extension
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if file has video extension
    """
    video_extensions = {'.mp4', '.webm'}
    return file_path.suffix.lower() in video_extensions

def validate_model_repo(repo_id: str) -> Dict[str, str]:
    """Validate HuggingFace model repository name
    
    Args:
        repo_id: Repository ID in format "username/model-name"
        
    Returns:
        Dict with error message if invalid, or None if valid
    """
    if not repo_id:
        return {"error": "Repository name is required"}
    
    if "/" not in repo_id:
        return {"error": "Repository name must be in format username/model-name"}
        
    # Check characters
    invalid_chars = set('<>:"/\\|?*')
    if any(c in repo_id for c in invalid_chars):
        return {"error": "Repository name contains invalid characters"}
        
    return {"error": None}

def save_to_hub(model_path: Path, repo_id: str, token: str, commit_message: str = "Update model") -> bool:
    """Save model files to Hugging Face Hub
    
    Args:
        model_path: Path to model files
        repo_id: Repository ID (username/model-name)
        token: HuggingFace API token
        commit_message: Optional commit message
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        api = HfApi(token=token)
        
        # Validate repo_id
        validation = validate_model_repo(repo_id)
        if validation["error"]:
            return False
        
        # Create or get repo
        try:
            create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"Error creating repo: {e}")
            return False
            
        # Upload all files
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        
        return True
    except Exception as e:
        print(f"Error uploading to hub: {e}")
        return False

def parse_training_log(line: str) -> Dict:
    """Parse a training log line for metrics
    
    Args:
        line: Log line from training output
        
    Returns:
        Dict with parsed metrics (epoch, step, loss, etc)
    """
    metrics = {}
    
    try:
        # Extract step/epoch info
        if "step=" in line:
            step = int(line.split("step=")[1].split()[0].strip(","))
            metrics["step"] = step
        
        if "epoch=" in line:
            epoch = int(line.split("epoch=")[1].split()[0].strip(","))
            metrics["epoch"] = epoch
            
        if "loss=" in line:
            loss = float(line.split("loss=")[1].split()[0].strip(","))
            metrics["loss"] = loss
            
        if "lr=" in line:
            lr = float(line.split("lr=")[1].split()[0].strip(","))
            metrics["learning_rate"] = lr
    except:
        pass
        
    return metrics

def format_size(size_bytes: int) -> str:
    """Format bytes into human readable string with appropriate unit
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g. "1.5 Gb")
    """
    units = ['bytes', 'Kb', 'Mb', 'Gb', 'Tb']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
        
    # Special case for bytes - no decimal places
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    
    return f"{size:.1f} {units[unit_index]}"


def count_media_files(path: Path) -> Tuple[int, int, int]:
    """Count videos and images in directory
    
    Args:
        path: Directory to scan
        
    Returns:
        Tuple of (video_count, image_count, total_size)
    """
    video_count = 0
    image_count = 0
    total_size = 0
    
    for file in path.glob("*"):
        # Skip hidden files and caption files
        if file.name.startswith('.') or file.suffix.lower() == '.txt':
            continue
            
        if is_video_file(file):
            video_count += 1
            total_size += file.stat().st_size
        elif is_image_file(file):
            image_count += 1
            total_size += file.stat().st_size
            
    return video_count, image_count, total_size

def format_media_title(action: str, video_count: int, image_count: int, total_size: int) -> str:
    """Format title with media counts and size
    
    Args:
        action: Action (eg "split", "caption")
        video_count: Number of videos
        image_count: Number of images
        total_size: Total size in bytes
        
    Returns:
        Formatted title string
    """
    parts = []
    if image_count > 0:
        parts.append(f"{image_count:,} photo{'s' if image_count != 1 else ''}")
    if video_count > 0:
        parts.append(f"{video_count:,} video{'s' if video_count != 1 else ''}")
        
    if not parts:
        return f"## 0 files to {action} (0 bytes)"
        
    return f"## {' and '.join(parts)} to {action} ({format_size(total_size)})"

def add_prefix_to_caption(caption: str, prefix: str) -> str:
    """Add prefix to caption if not already present"""
    if not prefix or not caption:
        return caption
    if caption.startswith(prefix):
        return caption
    return f"{prefix}{caption}"

def format_time(seconds: float) -> str:
    """Format time duration in seconds to human readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g. "2h 30m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
        
    return " ".join(parts)
import cv2
import numpy as np
from pathlib import Path
import subprocess

def detect_black_bars(video_path: Path) -> tuple[int, int, int, int]:
    """Detect black bars in video by analyzing first few frames
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (top, bottom, left, right) crop values
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Read first few frames to get stable detection
    frames_to_check = 5
    frames = []
    
    for _ in range(frames_to_check):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise ValueError("Could not read any frames from video")
    
    # Convert frames to grayscale and find average
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    avg_frame = np.mean(gray_frames, axis=0)
    
    # Threshold to detect black regions (adjust sensitivity if needed)
    threshold = 20
    black_mask = avg_frame < threshold
    
    # Find black bars by analyzing row/column means
    row_means = np.mean(black_mask, axis=1)
    col_means = np.mean(black_mask, axis=0)
    
    # Detect edges where black bars end (using high threshold to avoid false positives)
    black_threshold = 0.95  # 95% of pixels in row/col must be black
    
    # Find top and bottom crops
    top_crop = 0
    bottom_crop = black_mask.shape[0]
    
    for i, mean in enumerate(row_means):
        if mean > black_threshold:
            top_crop = i + 1
        else:
            break
            
    for i, mean in enumerate(reversed(row_means)):
        if mean > black_threshold:
            bottom_crop = black_mask.shape[0] - i - 1
        else:
            break
    
    # Find left and right crops
    left_crop = 0
    right_crop = black_mask.shape[1]
    
    for i, mean in enumerate(col_means):
        if mean > black_threshold:
            left_crop = i + 1
        else:
            break
            
    for i, mean in enumerate(reversed(col_means)):
        if mean > black_threshold:
            right_crop = black_mask.shape[1] - i - 1
        else:
            break
    
    return top_crop, bottom_crop, left_crop, right_crop

def remove_black_bars(input_path: Path, output_path: Path) -> bool:
    """Remove black bars from video using FFmpeg
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        
    Returns:
        bool: True if successful, False if no cropping needed
    """
    try:
        # Detect black bars
        top, bottom, left, right = detect_black_bars(input_path)
        
        # Get video dimensions using OpenCV
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # If no significant black bars detected, return False
        if top < 10 and bottom > height - 10 and \
           left < 10 and right > width - 10:
            return False
        
        # Calculate crop dimensions
        crop_height = bottom - top
        crop_width = right - left
        
        if crop_height <= 0 or crop_width <= 0:
            return False
        
        # Use FFmpeg to crop and save video
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-vf', f'crop={crop_width}:{crop_height}:{left}:{top}',
            '-c:a', 'copy',  # Copy audio stream
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return True
        
    except Exception as e:
        print(f"Error removing black bars: {e}")
        return False
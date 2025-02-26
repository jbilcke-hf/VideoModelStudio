import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pillow_avif
import logging
from config import NORMALIZE_IMAGES_TO, JPEG_QUALITY

logger = logging.getLogger(__name__)

def normalize_image(input_path: Path, output_path: Path) -> bool:
    """Convert image to normalized format (PNG or JPEG) and optionally remove black bars
    
    Args:
        input_path: Source image path
        output_path: Target path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open image with PIL
        with Image.open(input_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img, mask=img.split()[1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy for black bar detection
            img_np = np.array(img)
            
            # Detect black bars
            top, bottom, left, right = detect_black_bars(img_np)
            
            # Crop if black bars detected
            if any([top > 0, bottom < img_np.shape[0] - 1, 
                   left > 0, right < img_np.shape[1] - 1]):
                img = img.crop((left, top, right, bottom))
            
            # Save as configured format
            if NORMALIZE_IMAGES_TO == 'png':
                img.save(output_path, 'PNG', optimize=True)
            else:  # jpg
                img.save(output_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)
            return True
            
    except Exception as e:
        logger.error(f"Error converting image {input_path}: {str(e)}")
        return False

def detect_black_bars(img: np.ndarray) -> tuple[int, int, int, int]:
    """Detect black bars in image
    
    Args:
        img: numpy array of image (HxWxC)
        
    Returns:
        Tuple of (top, bottom, left, right) crop coordinates
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    # Threshold to detect black regions
    threshold = 20
    black_mask = gray < threshold
    
    # Find black bars by analyzing row/column means
    row_means = np.mean(black_mask, axis=1)
    col_means = np.mean(black_mask, axis=0)
    
    # Detect edges where black bars end (95% threshold)
    black_threshold = 0.95
    
    # Find top and bottom crops
    top = 0
    bottom = img.shape[0]
    
    for i, mean in enumerate(row_means):
        if mean > black_threshold:
            top = i + 1
        else:
            break
            
    for i, mean in enumerate(reversed(row_means)):
        if mean > black_threshold:
            bottom = img.shape[0] - i - 1
        else:
            break
    
    # Find left and right crops
    left = 0
    right = img.shape[1]
    
    for i, mean in enumerate(col_means):
        if mean > black_threshold:
            left = i + 1
        else:
            break
            
    for i, mean in enumerate(reversed(col_means)):
        if mean > black_threshold:
            right = img.shape[1] - i - 1
        else:
            break
            
    return top, bottom, left, right


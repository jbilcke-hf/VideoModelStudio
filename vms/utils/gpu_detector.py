import torch
import logging

logger = logging.getLogger(__name__)

def get_available_gpu_count():
    """Get the number of available GPUs on the system.
    
    Returns:
        int: Number of available GPUs, or 0 if no GPUs are available
    """
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    except Exception as e:
        logger.warning(f"Error detecting GPUs: {e}")
        return 0

def get_gpu_info():
    """Get information about available GPUs.
    
    Returns:
        list: List of dictionaries with GPU information
    """
    gpu_info = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory
                }
                gpu_info.append(gpu)
    except Exception as e:
        logger.warning(f"Error getting GPU details: {e}")
    
    return gpu_info

def get_recommended_precomputation_items(num_videos, num_gpus):
    """Calculate recommended precomputation items.
    
    Args:
        num_videos (int): Number of videos in dataset
        num_gpus (int): Number of GPUs to use
    
    Returns:
        int: Recommended precomputation items value
    """
    if num_gpus <= 0:
        num_gpus = 1
    
    # Calculate items per GPU, but ensure it's at least 1
    items_per_gpu = max(1, num_videos // num_gpus)
    
    # Limit to a maximum of 512
    return min(512, items_per_gpu)
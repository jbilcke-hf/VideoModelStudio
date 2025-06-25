"""
Monkey patch for Finetrainers to support loading existing LoRA weights as training initialization.

This patch extends the SFTTrainer to accept a --pretrained_lora_path argument that allows
starting training from existing LoRA weights instead of random initialization.
"""

import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path

import safetensors.torch
from peft import set_peft_model_state_dict

logger = logging.getLogger(__name__)

# Global flag to track if patch has been applied
_PATCH_APPLIED = False

def _load_pretrained_lora_weights(self, lora_path: str) -> None:
    """Load existing LoRA weights as training initialization
    
    Args:
        lora_path: Path to directory containing pytorch_lora_weights.safetensors
    """
    lora_path = Path(lora_path)
    
    # Find the safetensors file
    safetensors_file = lora_path / "pytorch_lora_weights.safetensors"
    if not safetensors_file.exists():
        raise FileNotFoundError(f"LoRA weights file not found: {safetensors_file}")
    
    logger.info(f"Loading pretrained LoRA weights from: {safetensors_file}")
    
    try:
        # Load the LoRA weights
        lora_state_dict = safetensors.torch.load_file(str(safetensors_file))
        
        # Extract metadata if available
        metadata = {}
        try:
            with open(safetensors_file, 'rb') as f:
                # Try to read metadata from safetensors header
                header_size = int.from_bytes(f.read(8), 'little')
                header_data = f.read(header_size)
                header = json.loads(header_data.decode('utf-8'))
                metadata = header.get('__metadata__', {})
        except Exception as e:
            logger.debug(f"Could not read metadata from safetensors: {e}")
        
        # Log metadata info if available
        if metadata:
            logger.info(f"LoRA metadata: rank={metadata.get('rank', 'unknown')}, "
                       f"alpha={metadata.get('lora_alpha', 'unknown')}")
        
        # Apply the LoRA weights to the model
        set_peft_model_state_dict(self.transformer, lora_state_dict)
        
        logger.info(f"Successfully loaded LoRA weights from {safetensors_file}")
        
        # Log the loaded keys for debugging
        logger.debug(f"Loaded LoRA keys: {list(lora_state_dict.keys())}")
        
    except Exception as e:
        logger.error(f"Failed to load LoRA weights from {safetensors_file}: {e}")
        raise RuntimeError(f"Failed to load LoRA weights: {e}")


def patched_prepare_trainable_parameters(self) -> None:
    """Patched version of _prepare_trainable_parameters that supports pretrained LoRA loading"""
    
    # Call the original method first
    original_prepare_trainable_parameters(self)
    
    # Check if pretrained LoRA path is provided
    if hasattr(self.args, 'pretrained_lora_path') and self.args.pretrained_lora_path:
        logger.info(f"Pretrained LoRA path specified: {self.args.pretrained_lora_path}")
        
        # Only load if we're doing LoRA training
        if hasattr(self.args, 'training_type') and str(self.args.training_type) == 'TrainingType.LORA':
            self._load_pretrained_lora_weights(self.args.pretrained_lora_path)
        else:
            logger.warning("pretrained_lora_path specified but training_type is not LORA")


def apply_lora_loading_patch() -> None:
    """Apply the monkey patch to enable LoRA weight loading in Finetrainers"""
    global _PATCH_APPLIED
    
    if _PATCH_APPLIED:
        logger.debug("Finetrainers LoRA loading patch already applied")
        return
    
    try:
        from finetrainers.trainer.sft_trainer.trainer import SFTTrainer
        
        # Store reference to original method
        global original_prepare_trainable_parameters
        original_prepare_trainable_parameters = SFTTrainer._prepare_trainable_parameters
        
        # Apply patches
        SFTTrainer._prepare_trainable_parameters = patched_prepare_trainable_parameters
        SFTTrainer._load_pretrained_lora_weights = _load_pretrained_lora_weights
        
        _PATCH_APPLIED = True
        logger.info("Successfully applied Finetrainers LoRA loading patch")
        
    except ImportError as e:
        logger.error(f"Failed to import Finetrainers classes for patching: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to apply Finetrainers LoRA loading patch: {e}")
        raise


def remove_lora_loading_patch() -> None:
    """Remove the monkey patch (for testing purposes)"""
    global _PATCH_APPLIED
    
    if not _PATCH_APPLIED:
        return
    
    try:
        from finetrainers.trainer.sft_trainer.trainer import SFTTrainer
        
        # Restore original method
        SFTTrainer._prepare_trainable_parameters = original_prepare_trainable_parameters
        
        # Remove added method
        if hasattr(SFTTrainer, '_load_pretrained_lora_weights'):
            delattr(SFTTrainer, '_load_pretrained_lora_weights')
        
        _PATCH_APPLIED = False
        logger.info("Removed Finetrainers LoRA loading patch")
        
    except Exception as e:
        logger.error(f"Failed to remove Finetrainers LoRA loading patch: {e}")


# Store reference to original method (will be set when patch is applied)
original_prepare_trainable_parameters = None
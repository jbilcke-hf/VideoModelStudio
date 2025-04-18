"""
Models service for Video Model Studio

Handles the model history tracking and management
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from vms.config import (
    STORAGE_PATH, MODEL_TYPES, TRAINING_TYPES
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class Model:
    """Class for tracking model metadata"""
    id: str
    status: str  # 'draft', 'training', 'trained', 'error'
    model_type: str  # Base model family (e.g. 'hunyuan_video', 'ltx_video', 'wan')
    model_display_name: str  # Display name for the model type
    created_at: datetime
    updated_at: datetime
    training_progress: Optional[float] = 0.0  # Progress as percentage
    current_step: Optional[int] = 0
    total_steps: Optional[int] = 0
    
    @classmethod
    def from_dir(cls, model_dir: Path) -> 'Model':
        """Create a Model instance from a directory"""
        model_id = model_dir.name
        
        # Default values
        status = 'draft'
        model_type = ''
        model_display_name = ''
        created_at = datetime.fromtimestamp(model_dir.stat().st_ctime)
        updated_at = datetime.fromtimestamp(model_dir.stat().st_mtime)
        training_progress = 0.0
        current_step = 0
        total_steps = 0
        
        # Check for UI state file
        ui_state_file = model_dir / "output" / "ui_state.json"
        if ui_state_file.exists():
            try:
                with open(ui_state_file, 'r') as f:
                    ui_state = json.load(f)
                    status = ui_state.get('project_status', 'draft')
                    
                    # Get model type from UI state
                    model_type_value = ui_state.get('model_type', '')
                    
                    # First check if model_type_value is a display name
                    display_name_found = False
                    for display_name, internal_name in MODEL_TYPES.items():
                        if display_name == model_type_value:
                            model_type = internal_name
                            model_display_name = display_name
                            display_name_found = True
                            break
                    
                    # If not a display name, check if it's an internal name
                    if not display_name_found:
                        for display_name, internal_name in MODEL_TYPES.items():
                            if internal_name == model_type_value:
                                model_type = internal_name
                                model_display_name = display_name
                                break
            except Exception as e:
                logger.error(f"Error loading UI state for model {model_id}: {str(e)}")
        
        # Check for status file to get training progress
        status_file = model_dir / "output" / "status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    if status_data.get('status') == 'training':
                        status = 'training'
                        current_step = status_data.get('step', 0)
                        total_steps = status_data.get('total_steps', 0)
                        if total_steps > 0:
                            training_progress = (current_step / total_steps) * 100
                    elif status_data.get('status') == 'completed':
                        status = 'trained'
                        training_progress = 100.0
                    elif status_data.get('status') == 'error':
                        status = 'error'
            except Exception as e:
                logger.error(f"Error loading status for model {model_id}: {str(e)}")
        
        # Check for pid file to determine if training is active
        pid_file = model_dir / "output" / "training.pid"
        if pid_file.exists():
            status = 'training'
        
        # Check for model weights to determine if trained
        model_weights = model_dir / "output" / "pytorch_lora_weights.safetensors"
        if model_weights.exists() and status != 'training':
            status = 'trained'
            training_progress = 100.0
        
        return cls(
            id=model_id,
            status=status,
            model_type=model_type,
            model_display_name=model_display_name,
            created_at=created_at,
            updated_at=updated_at,
            training_progress=training_progress,
            current_step=current_step,
            total_steps=total_steps
        )

class ModelsService:
    """Service for tracking and managing model history"""
    
    def __init__(self, app_state=None):
        """Initialize the models service
        
        Args:
            app_state: Reference to main application state
        """
        self.app = app_state
    
    def get_all_models(self) -> List[Model]:
        """Get a list of all models
        
        Returns:
            List of Model objects
        """
        models_dir = STORAGE_PATH / "models"
        if not models_dir.exists():
            return []
            
        models = []
        
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            try:
                model = Model.from_dir(model_dir)
                models.append(model)
            except Exception as e:
                logger.error(f"Error loading model from {model_dir}: {str(e)}")
                
        # Sort models by updated_at (newest first)
        return sorted(models, key=lambda m: m.updated_at, reverse=True)
    
    def get_draft_models(self) -> List[Model]:
        """Get a list of draft models
        
        Returns:
            List of Model objects with 'draft' status
        """
        return [m for m in self.get_all_models() if m.status == 'draft']
    
    def get_training_models(self) -> List[Model]:
        """Get a list of models currently in training
        
        Returns:
            List of Model objects with 'training' status
        """
        return [m for m in self.get_all_models() if m.status == 'training']
    
    def get_trained_models(self) -> List[Model]:
        """Get a list of completed trained models
        
        Returns:
            List of Model objects with 'trained' status
        """
        return [m for m in self.get_all_models() if m.status == 'trained']
        
    def delete_model(self, model_id: str) -> bool:
        """Delete a model by ID
        
        Args:
            model_id: The model ID to delete
            
        Returns:
            True if deletion was successful
        """
        if not model_id:
            return False
            
        model_dir = STORAGE_PATH / "models" / model_id
        if not model_dir.exists():
            return False
            
        try:
            import shutil
            shutil.rmtree(model_dir)
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False
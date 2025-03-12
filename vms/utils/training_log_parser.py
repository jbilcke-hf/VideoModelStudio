import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Represents the current state of training"""
    status: str = "idle"  # idle, initializing, training, completed, error, stopped
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    step_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    memory_allocated: float = 0.0
    memory_reserved: float = 0.0
    start_time: Optional[datetime] = None
    last_step_time: Optional[datetime] = None
    estimated_remaining: Optional[str] = None
    error_message: Optional[str] = None
    initialization_stage: str = ""
    download_progress: float = 0.0
    elapsed_time: str = "0:00:00"
    
    # New fields for current task tracking
    current_task: str = ""
    current_task_progress: str = ""
    task_progress_percentage: float = 0.0
    task_items_processed: int = 0
    task_total_items: int = 0
    task_time_remaining: str = ""
    task_speed: str = ""
    
    # Store recent progress lines for task display
    recent_progress_lines: List[str] = None

    def __post_init__(self):
        if self.recent_progress_lines is None:
            self.recent_progress_lines = []

    def calculate_progress(self) -> float:
        """Calculate overall progress as percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for UI updates"""
        # Use the stored elapsed time directly if it exists
        elapsed = self.elapsed_time
        
        # Use precomputed remaining time from logs if available
        remaining = str(self.estimated_remaining) if self.estimated_remaining else "calculating..."
        
        result = {
            "status": self.status,
            "progress": f"{self.calculate_progress():.1f}%",
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "step_loss": f"{self.step_loss:.4f}",
            "learning_rate": f"{self.learning_rate:.2e}",
            "grad_norm": f"{self.grad_norm:.4f}",
            "memory": f"{self.memory_allocated:.1f}GB allocated, {self.memory_reserved:.1f}GB reserved",
            "elapsed": elapsed,
            "remaining": remaining,
            "initialization_stage": self.initialization_stage,
            "error_message": self.error_message,
            "download_progress": self.download_progress
        }
        
        # Add current task information
        result["current_task"] = self.get_task_display()
        
        return result
    
    def get_task_display(self) -> str:
        """Generate a formatted display of the current task"""
        if not self.recent_progress_lines:
            if self.status == "training":
                return "Training in progress..."
            return ""
        
        # Get the most recent progress line
        latest_line = self.recent_progress_lines[-1]
        
        # For downloading shards or loading checkpoint shards
        if "Downloading shards" in latest_line or "Loading checkpoint shards" in latest_line:
            # Extract just the progress bar part
            match = re.search(r'(\d+%\|[▏▎▍▌▋▊▉█\s]+\|)', latest_line)
            if match:
                progress_bar = match.group(1)
                
                # Extract the remaining information
                time_match = re.search(r'\[(\d+:\d+<\d+:\d+,\s+[\d.]+s/it)', latest_line)
                time_info = time_match.group(1) if time_match else ""
                
                task_type = "Downloading shards" if "Downloading shards" in latest_line else "Loading checkpoint shards"
                
                return f"{task_type}:\n{progress_bar}\n{time_info}"
        
        # For "Rank 0" progress (typically training steps)
        elif "Rank 0:" in latest_line:
            match = re.search(r'Rank 0:\s+(\d+%\|[▏▎▍▌▋▊▉█\s]+\|)', latest_line)
            if match:
                progress_bar = match.group(1)
                
                # Extract step information
                step_match = re.search(r'\|\s+(\d+/\d+)', latest_line)
                step_info = step_match.group(1) if step_match else ""
                
                # Extract time information
                time_match = re.search(r'\[(\d+:\d+<\d+:\d+,\s+[\d.]+s/it)', latest_line)
                time_info = time_match.group(1) if time_match else ""
                
                return f"Training iteration:\n{progress_bar} {step_info}\n{time_info}"
        
        # For Filling buffer progress
        elif "Filling buffer" in latest_line:
            match = re.search(r'(\d+%\|[▏▎▍▌▋▊▉█\s]+\|)', latest_line)
            if match:
                progress_bar = match.group(1)
                
                # Extract step information
                step_match = re.search(r'\|\s+(\d+/\d+)', latest_line)
                step_info = step_match.group(1) if step_match else ""
                
                # Extract time information
                time_match = re.search(r'\[(\d+:\d+<\d+:\d+,\s+[\d.]+s/it)', latest_line)
                time_info = time_match.group(1) if time_match else ""
                
                return f"Filling buffer from data iterator:\n{progress_bar} {step_info}\n{time_info}"
                
        # For other progress lines
        elif "%" in latest_line and "|" in latest_line:
            # Generic progress bar pattern
            match = re.search(r'(\d+%\|[▏▎▍▌▋▊▉█\s]+\|)', latest_line)
            if match:
                progress_bar = match.group(1)
                
                # Try to extract step information
                step_match = re.search(r'\|\s+(\d+/\d+)', latest_line)
                step_info = step_match.group(1) if step_match else ""
                
                # Try to extract time information
                time_match = re.search(r'\[(\d+:\d+<\d+:\d+,\s+[\d.]+s/it)', latest_line)
                time_info = time_match.group(1) if time_match else ""
                
                task_prefix = "Processing:"
                
                # Try to determine task type
                if "Training" in latest_line:
                    task_prefix = "Training:"
                elif "Precomputing" in latest_line:
                    task_prefix = "Precomputing:"
                
                return f"{task_prefix}\n{progress_bar} {step_info}\n{time_info}"
        
        # If we couldn't parse it properly, just return the line
        return latest_line.strip()

class TrainingLogParser:
    """Parser for training logs with state management"""
    
    def __init__(self):
        self.state = TrainingState()
        self._last_update_time = None
        # Maximum number of recent progress lines to store
        self.max_recent_lines = 5
        
    def reset(self):
        """Reset parser state"""
        self.state = TrainingState()
        self._last_update_time = None
    
    def get_current_task_display(self) -> str:
        """Get the formatted current task display"""
        return self.state.get_task_display()
    
    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line and update state"""
        try:
            # Check if this is a progress line
            if any(pattern in line for pattern in ["Downloading shards:", "Loading checkpoint shards:", "Rank 0:", "Filling buffer", "|"]) and "%" in line:
                # Add to recent progress lines, maintaining order and max length
                self.state.recent_progress_lines.append(line)
                if len(self.state.recent_progress_lines) > self.max_recent_lines:
                    self.state.recent_progress_lines.pop(0)
                
                # Parse the Training steps line for additional information
                if "Training steps:" in line:
                    # Set status to training if we see this
                    self.state.status = "training"
                    
                    if not self.state.start_time:
                        self.state.start_time = datetime.now()

                    # Extract step numbers from the format: Training steps:   4%|▍         | 44/1000 [41:57<17:22:32, 65.43s/it]
                    steps_match = re.search(r"\|\s*(\d+)/(\d+)", line)
                    if steps_match:
                        self.state.current_step = int(steps_match.group(1))
                        self.state.total_steps = int(steps_match.group(2))
                    
                    # Extract elapsed time - Format example: [41:57<17:22:32, 65.43s/it]
                    elapsed_match = re.search(r"\[(\d+:\d+)(:\d+)?<", line)
                    if elapsed_match:
                        if elapsed_match.group(2):  # has hours:minutes:seconds format
                            self.state.elapsed_time = elapsed_match.group(1) + elapsed_match.group(2)
                        else:  # has minutes:seconds format
                            self.state.elapsed_time = elapsed_match.group(1)
                    
                    # Extract remaining time - Format example: [41:57<17:22:32, 65.43s/it]
                    remaining_match = re.search(r"<([\d:]+)", line)
                    if remaining_match:
                        self.state.estimated_remaining = remaining_match.group(1)
                    
                    # Extract metrics with different patterns
                    # Pattern 1: grad_norm=0.113, global_avg_loss=0.15, global_max_loss=0.15
                    grad_norm_match = re.search(r"grad_norm=([0-9.e-]+)", line)
                    if grad_norm_match:
                        self.state.grad_norm = float(grad_norm_match.group(1))
                    
                    # Try global_avg_loss as the main loss metric
                    loss_match = re.search(r"global_avg_loss=([0-9.e-]+)", line)
                    if loss_match:
                        self.state.step_loss = float(loss_match.group(1))
                    elif "step_loss=" in line:
                        # Fall back to step_loss if global_avg_loss not found
                        loss_match = re.search(r"step_loss=([0-9.e-]+)", line)
                        if loss_match:
                            self.state.step_loss = float(loss_match.group(1))
                    
                    # Extract learning rate if available
                    lr_match = re.search(r"lr=([0-9.e-]+)", line)
                    if lr_match:
                        self.state.learning_rate = float(lr_match.group(1))
                    
                    # Update last processing time
                    self.state.last_step_time = datetime.now()
                
                # Return updated state
                return self.state.to_dict()
            
            # Parse "Starting training step" lines to extract step/total info if not already parsed
            step_match = re.search(r"Starting training step \((\d+)/(\d+)\)", line)
            if step_match:
                current_step = int(step_match.group(1))
                total_steps = int(step_match.group(2))
                
                # Only update if we don't already have a value or if this is more recent
                if self.state.total_steps == 0 or current_step > self.state.current_step:
                    self.state.current_step = current_step
                    self.state.total_steps = total_steps
                    self.state.status = "training"  # Ensure status is set to training
                    logger.info(f"Updated training step: {current_step}/{total_steps}")
                    return self.state.to_dict()

            if ("Started training" in line) or ("Starting training" in line):
                self.state.status = "training"
                if not self.state.start_time:
                    self.state.start_time = datetime.now()
                return self.state.to_dict()

            # Epoch information
            epoch_match = re.search(r"Starting epoch \((\d+)/(\d+)\)", line)
            if epoch_match:
                self.state.current_epoch = int(epoch_match.group(1))
                self.state.total_epochs = int(epoch_match.group(2))
                logger.info(f"Updated epoch: {self.state.current_epoch}/{self.state.total_epochs}")
                return self.state.to_dict()

            # Initialization stages
            if "Initializing" in line:
                self.state.status = "initializing"
                self.state.initialization_stage = line.split("Initializing")[1].strip()
                logger.info(f"Initialization stage: {self.state.initialization_stage}")
                return self.state.to_dict()

            # Memory usage
            if "memory_allocated" in line:
                mem_match = re.search(r'"memory_allocated":\s*([0-9.]+)', line)
                if mem_match:
                    self.state.memory_allocated = float(mem_match.group(1))
                
                reserved_match = re.search(r'"memory_reserved":\s*([0-9.]+)', line)
                if reserved_match:
                    self.state.memory_reserved = float(reserved_match.group(1))
                logger.info(f"Updated memory: allocated={self.state.memory_allocated}GB, reserved={self.state.memory_reserved}GB")
                return self.state.to_dict()

            # Completion states
            if "Training completed successfully" in line:
                self.state.status = "completed"
                # Store final elapsed time
                self.state.last_step_time = datetime.now()
                logger.info("Training completed")
                return self.state.to_dict()

            if any(x in line for x in ["Training process stopped", "Training stopped"]):
                self.state.status = "stopped"
                # Store final elapsed time
                self.state.last_step_time = datetime.now()
                logger.info("Training stopped")
                return self.state.to_dict()

            if "Error during training:" in line:
                self.state.status = "error"
                self.state.error_message = line.split("Error during training:")[1].strip()
                logger.info(f"Training error: {self.state.error_message}")
                return self.state.to_dict()

        except Exception as e:
            logger.error(f"Error parsing line: {str(e)}")
            
        return None
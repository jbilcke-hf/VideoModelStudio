import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
    estimated_remaining: Optional[timedelta] = None
    error_message: Optional[str] = None
    initialization_stage: str = ""
    download_progress: float = 0.0

    def calculate_progress(self) -> float:
        """Calculate overall progress as percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for UI updates"""
        # Calculate elapsed time only if training is active and we have a start time
        if self.start_time and self.status in ["training", "initializing"]:
            elapsed = str(datetime.now() - self.start_time)
        else:
            # Use the last known elapsed time or show 0
            elapsed = "0:00:00" if not self.last_step_time else str(self.last_step_time - self.start_time if self.start_time else "0:00:00")
        
        # Use precomputed remaining time from logs if available
        remaining = str(self.estimated_remaining) if self.estimated_remaining else "calculating..."
        
        return {
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

class TrainingLogParser:
    """Parser for training logs with state management"""
    
    def __init__(self):
        self.state = TrainingState()
        self._last_update_time = None
        
    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line and update state"""
        try:
            # For debugging
            #logger.info(f"Parsing line: {line[:100]}...")

            # Training step progress line example:
            # Training steps:   1%|▏         | 1/70 [00:14<16:11, 14.08s/it, grad_norm=0.00789, step_loss=0.555, lr=3e-7]
            
            if ("Started training" in line) or ("Starting training" in line):
                self.state.status = "training"
            
            # Check for "Training steps:" which contains the progress information
            if "Training steps:" in line:
                # Set status to training if we see this
                self.state.status = "training"
                
                if not self.state.start_time:
                    self.state.start_time = datetime.now()

                # Extract step numbers
                steps_match = re.search(r"(\d+)/(\d+)", line)
                if steps_match:
                    self.state.current_step = int(steps_match.group(1))
                    self.state.total_steps = int(steps_match.group(2))

                # Extract metrics
                for pattern, attr in [
                    (r"step_loss=([0-9.e-]+)", "step_loss"),
                    (r"lr=([0-9.e-]+)", "learning_rate"),
                    (r"grad_norm=([0-9.e-]+)", "grad_norm")
                ]:
                    match = re.search(pattern, line)
                    if match:
                        setattr(self.state, attr, float(match.group(1)))

                # Extract time remaining directly from the log
                # Format: [MM:SS<M:SS:SS, SS.SSs/it]
                time_remaining_match = re.search(r"<(\d+:\d+:\d+)", line)
                if time_remaining_match:
                    remaining_str = time_remaining_match.group(1)
                    # Store the string directly - no need to parse it
                    self.state.estimated_remaining = remaining_str
                
                # If no direct time estimate, look for hour:min format
                if not time_remaining_match:
                    hour_min_match = re.search(r"<(\d+h\s*\d+m)", line)
                    if hour_min_match:
                        self.state.estimated_remaining = hour_min_match.group(1)

                # Update last processing time
                self.state.last_step_time = datetime.now()
                
                logger.info(f"Updated training state: step={self.state.current_step}/{self.state.total_steps}, loss={self.state.step_loss}")
                return self.state.to_dict()

            # Epoch information
            # there is an issue with how epoch is reported because we display:
            # Progress: 96.9%, Step: 872/900, Epoch: 12/50
            # we should probably just show the steps
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
                self.status = "completed"
                # Store final elapsed time
                self.last_step_time = datetime.now()
                logger.info("Training completed")
                return self.state.to_dict()

            if any(x in line for x in ["Training process stopped", "Training stopped"]):
                self.status = "stopped"
                # Store final elapsed time
                self.last_step_time = datetime.now()
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

    def reset(self):
        """Reset parser state"""
        self.state = TrainingState()
        self._last_update_time = None
import os
import sys
import json
import time
import shutil
import gradio as gr
from pathlib import Path
from datetime import datetime
import subprocess
import signal
import psutil
import tempfile
import zipfile
import logging
import traceback
import threading
import select

from typing import Any, Optional, Dict, List, Union, Tuple

from huggingface_hub import upload_folder, create_repo

from ..config import (
    TrainingConfig, TRAINING_PRESETS, LOG_FILE_PATH, TRAINING_VIDEOS_PATH, 
    STORAGE_PATH, TRAINING_PATH, MODEL_PATH, OUTPUT_PATH, HF_API_TOKEN, 
    MODEL_TYPES, TRAINING_TYPES
)
from ..utils import make_archive, parse_training_log, is_image_file, is_video_file, prepare_finetrainers_dataset, copy_files_to_training_dir

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        # State and log files
        self.session_file = OUTPUT_PATH / "session.json"
        self.status_file = OUTPUT_PATH / "status.json"
        self.pid_file = OUTPUT_PATH / "training.pid"
        self.log_file = OUTPUT_PATH / "training.log"

        self.file_handler = None
        self.setup_logging()
        self.ensure_valid_ui_state_file()

        logger.info("Training service initialized")

    def setup_logging(self):
        """Set up logging with proper handler management"""
        global logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Add stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(stdout_handler)
        
        # Add file handler if log file is accessible
        try:
            # Close existing file handler if it exists
            if self.file_handler:
                self.file_handler.close()
                logger.removeHandler(self.file_handler)
            
            self.file_handler = logging.FileHandler(str(LOG_FILE_PATH))
            self.file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(self.file_handler)
        except Exception as e:
            logger.warning(f"Could not set up log file: {e}")

    def clear_logs(self) -> None:
        """Clear log file with proper handler cleanup"""
        try:
            # Remove and close the file handler
            if self.file_handler:
                logger.removeHandler(self.file_handler)
                self.file_handler.close()
                self.file_handler = None
            
            # Delete the file if it exists
            if LOG_FILE_PATH.exists():
                LOG_FILE_PATH.unlink()
            
            # Recreate logging setup
            self.setup_logging()
            self.append_log("Log file cleared and recreated")
            
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")
            raise
    
    def __del__(self):
        """Cleanup when the service is destroyed"""
        if self.file_handler:
            self.file_handler.close()
    
            
    def save_ui_state(self, values: Dict[str, Any]) -> None:
        """Save current UI state to file"""
        ui_state_file = OUTPUT_PATH / "ui_state.json"
        try:
            with open(ui_state_file, 'w') as f:
                json.dump(values, f, indent=2)
            logger.debug(f"UI state saved: {values}")
        except Exception as e:
            logger.error(f"Error saving UI state: {str(e)}")

    def load_ui_state(self) -> Dict[str, Any]:
        """Load saved UI state"""
        ui_state_file = OUTPUT_PATH / "ui_state.json"
        default_state = {
            "model_type": list(MODEL_TYPES.keys())[0],
            "training_type": list(TRAINING_TYPES.keys())[0],
            "lora_rank": "128",
            "lora_alpha": "128", 
            "num_epochs": 50,
            "batch_size": 1,
            "learning_rate": 3e-5,
            "save_iterations": 200,
            "training_preset": list(TRAINING_PRESETS.keys())[0]
        }
        
        if not ui_state_file.exists():
            logger.info("UI state file does not exist, using default values")
            return default_state
                
        try:
            # First check if the file is empty
            file_size = ui_state_file.stat().st_size
            if file_size == 0:
                logger.warning("UI state file exists but is empty, using default values")
                return default_state
                
            with open(ui_state_file, 'r') as f:
                file_content = f.read().strip()
                if not file_content:
                    logger.warning("UI state file is empty or contains only whitespace, using default values")
                    return default_state
                    
                saved_state = json.loads(file_content)
                
                # Convert numeric values to appropriate types
                if "num_epochs" in saved_state:
                    saved_state["num_epochs"] = int(saved_state["num_epochs"])
                if "batch_size" in saved_state:
                    saved_state["batch_size"] = int(saved_state["batch_size"])
                if "learning_rate" in saved_state:
                    saved_state["learning_rate"] = float(saved_state["learning_rate"])
                if "save_iterations" in saved_state:
                    saved_state["save_iterations"] = int(saved_state["save_iterations"])
                    
                # Make sure we have all keys (in case structure changed)
                merged_state = default_state.copy()
                merged_state.update(saved_state)
                return merged_state
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing UI state JSON: {str(e)}")
            return default_state
        except Exception as e:
            logger.error(f"Error loading UI state: {str(e)}")
            return default_state

    def ensure_valid_ui_state_file(self):
        """Ensure UI state file exists and is valid JSON"""
        ui_state_file = OUTPUT_PATH / "ui_state.json"
        
        if not ui_state_file.exists():
            # Create a new file with default values
            logger.info("Creating new UI state file with default values")
            default_state = {
                "model_type": list(MODEL_TYPES.keys())[0],
                "training_type": list(TRAINING_TYPES.keys())[0],
                "lora_rank": "128",
                "lora_alpha": "128", 
                "num_epochs": 50,
                "batch_size": 1,
                "learning_rate": 3e-5,
                "save_iterations": 200,
                "training_preset": list(TRAINING_PRESETS.keys())[0]
            }
            self.save_ui_state(default_state)
            return
        
        # Check if file is valid JSON
        try:
            with open(ui_state_file, 'r') as f:
                file_content = f.read().strip()
                if not file_content:
                    raise ValueError("Empty file")
                json.loads(file_content)
            logger.debug("UI state file validation successful")
        except Exception as e:
            logger.warning(f"Invalid UI state file detected: {str(e)}. Creating new one with defaults.")
            # Backup the invalid file
            backup_file = ui_state_file.with_suffix('.json.bak')
            try:
                shutil.copy2(ui_state_file, backup_file)
                logger.info(f"Backed up invalid UI state file to {backup_file}")
            except Exception as backup_error:
                logger.error(f"Failed to backup invalid UI state file: {str(backup_error)}")
            
            # Create a new file with default values
            default_state = {
                "model_type": list(MODEL_TYPES.keys())[0],
                "training_type": list(TRAINING_TYPES.keys())[0],
                "lora_rank": "128",
                "lora_alpha": "128", 
                "num_epochs": 50,
                "batch_size": 1,
                "learning_rate": 3e-5,
                "save_iterations": 200,
                "training_preset": list(TRAINING_PRESETS.keys())[0]
            }
            self.save_ui_state(default_state)
            
    # Modify save_session to also store the UI state at training start
    def save_session(self, params: Dict) -> None:
        """Save training session parameters"""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "status": self.get_status(),
            # Add UI state at the time training started
            "initial_ui_state": self.load_ui_state()
        }
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session(self) -> Optional[Dict]:
        """Load saved training session"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None

    def get_status(self) -> Dict:
        """Get current training status"""
        default_status = {'status': 'stopped', 'message': 'No training in progress'}
        
        if not self.status_file.exists():
            return default_status
                
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
                    
            # Check if process is actually running
            if self.pid_file.exists():
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                if not psutil.pid_exists(pid):
                    # Process died unexpectedly
                    if status['status'] == 'training':
                        # Only log this once by checking if we've already updated the status
                        if not hasattr(self, '_process_terminated_logged') or not self._process_terminated_logged:
                            self.append_log("Training process terminated unexpectedly")
                            self._process_terminated_logged = True
                        status['status'] = 'error'
                        status['message'] = 'Training process terminated unexpectedly'
                        # Update the status file to avoid repeated logging
                        with open(self.status_file, 'w') as f:
                            json.dump(status, f, indent=2)
                    else:
                        status['status'] = 'stopped'
                        status['message'] = 'Training process not found'
            return status
                
        except (json.JSONDecodeError, ValueError):
            return default_status

    def get_logs(self, max_lines: int = 100) -> str:
        """Get training logs with line limit"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                return ''.join(lines[-max_lines:])
        return ""

    def append_log(self, message: str) -> None:
        """Append message to log file and logger"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        logger.info(message)

    def clear_logs(self) -> None:
        """Clear log file"""
        if self.log_file.exists():
            self.log_file.unlink()
        self.append_log("Log file cleared")

    def validate_training_config(self, config: TrainingConfig, model_type: str) -> Optional[str]:
        """Validate training configuration"""
        logger.info(f"Validating config for {model_type}")
        
        try:
            # Basic validation
            if not config.output_dir:
                return "Output directory not specified"
                
            # For the dataset_config validation, we now expect it to be a JSON file
            dataset_config_path = Path(config.data_root)
            if not dataset_config_path.exists():
                return f"Dataset config file does not exist: {dataset_config_path}"
            
            # Check the JSON file is valid
            try:
                with open(dataset_config_path, 'r') as f:
                    dataset_json = json.load(f)
                
                # Basic validation of the JSON structure
                if "datasets" not in dataset_json or not isinstance(dataset_json["datasets"], list) or len(dataset_json["datasets"]) == 0:
                    return "Invalid dataset config JSON: missing or empty 'datasets' array"
                    
            except json.JSONDecodeError:
                return f"Invalid JSON in dataset config file: {dataset_config_path}"
            except Exception as e:
                return f"Error reading dataset config file: {str(e)}"
                    
            # Check training videos directory exists
            if not TRAINING_VIDEOS_PATH.exists():
                return f"Training videos directory does not exist: {TRAINING_VIDEOS_PATH}"
                
            # Validate file counts
            video_count = len(list(TRAINING_VIDEOS_PATH.glob('*.mp4')))
            
            if video_count == 0:
                return "No training files found"
                    
            # Model-specific validation
            if model_type == "hunyuan_video":
                if config.batch_size > 2:
                    return "Hunyuan model recommended batch size is 1-2"
                if not config.gradient_checkpointing:
                    return "Gradient checkpointing is required for Hunyuan model"
            elif model_type == "ltx_video":
                if config.batch_size > 4:
                    return "LTX model recommended batch size is 1-4"
            elif model_type == "wan":
                if config.batch_size > 4:
                    return "Wan model recommended batch size is 1-4"
                    
            logger.info(f"Config validation passed with {video_count} training files")
            return None
            
        except Exception as e:
            logger.error(f"Error during config validation: {str(e)}")
            return f"Configuration validation failed: {str(e)}"
        
    def start_training(
        self,
        model_type: str,
        lora_rank: str,
        lora_alpha: str,
        num_epochs: int,
        batch_size: int, 
        learning_rate: float,
        save_iterations: int,
        repo_id: str,
        preset_name: str,
        training_type: str = "lora",
        resume_from_checkpoint: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Start training with finetrainers"""
        
        self.clear_logs()

        if not model_type:
            raise ValueError("model_type cannot be empty")
        if model_type not in MODEL_TYPES.values():
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(MODEL_TYPES.values())}")
        if training_type not in TRAINING_TYPES.values():
            raise ValueError(f"Invalid training_type: {training_type}. Must be one of {list(TRAINING_TYPES.values())}")

        # Check if we're resuming or starting new
        is_resuming = resume_from_checkpoint is not None
        log_prefix = "Resuming" if is_resuming else "Initializing"
        logger.info(f"{log_prefix} training with model_type={model_type}, training_type={training_type}")
        
        try:
            # Get absolute paths - FIXED to look in project root instead of within vms directory
            current_dir = Path(__file__).parent.parent.parent.absolute()  # Go up to project root
            train_script = current_dir / "train.py"
            
            if not train_script.exists():
                # Try alternative locations
                alt_locations = [
                    current_dir.parent / "train.py",  # One level up from project root
                    Path("/home/user/app/train.py"),  # Absolute path
                    Path("train.py")  # Current working directory
                ]
                
                for alt_path in alt_locations:
                    if alt_path.exists():
                        train_script = alt_path
                        logger.info(f"Found train.py at alternative location: {train_script}")
                        break
                
                if not train_script.exists():
                    error_msg = f"Training script not found at {train_script} or any alternative locations"
                    logger.error(error_msg)
                    return error_msg, "Training script not found"
                    
            # Log paths for debugging
            logger.info("Current working directory: %s", current_dir)
            logger.info("Training script path: %s", train_script)
            logger.info("Training data path: %s", TRAINING_PATH)
            
            videos_file, prompts_file = prepare_finetrainers_dataset()
            if videos_file is None or prompts_file is None:
                error_msg = "Failed to generate training lists"
                logger.error(error_msg)
                return error_msg, "Training preparation failed"

            video_count = sum(1 for _ in open(videos_file))
            logger.info(f"Generated training lists with {video_count} files")

            if video_count == 0:
                error_msg = "No training files found"
                logger.error(error_msg)
                return error_msg, "No training data available"

            # Get preset configuration
            preset = TRAINING_PRESETS[preset_name]
            training_buckets = preset["training_buckets"]
            flow_weighting_scheme = preset.get("flow_weighting_scheme", "none")
            preset_training_type = preset.get("training_type", "lora")

            # Create a proper dataset configuration JSON file
            dataset_config_file = OUTPUT_PATH / "dataset_config.json"

            # Determine appropriate ID token based on model type
            id_token = None
            if model_type == "hunyuan_video":
                id_token = "afkx"
            elif model_type == "ltx_video":
                id_token = "BW_STYLE"
            # Wan doesn't use an ID token by default, so leave it as None

            dataset_config = {
                "datasets": [
                    {
                        "data_root": str(TRAINING_PATH),
                        "dataset_type": "video",
                        "id_token": id_token,
                        "video_resolution_buckets": [[f, h, w] for f, h, w in training_buckets],
                        "reshape_mode": "bicubic",
                        "remove_common_llm_caption_prefixes": True
                    }
                ]
            }

            # Write the dataset config to file
            with open(dataset_config_file, 'w') as f:
                json.dump(dataset_config, f, indent=2)

            logger.info(f"Created dataset configuration file at {dataset_config_file}")

            # Get config for selected model type with preset buckets
            if model_type == "hunyuan_video":
                if training_type == "lora":
                    config = TrainingConfig.hunyuan_video_lora(
                        data_path=str(TRAINING_PATH),
                        output_path=str(OUTPUT_PATH),
                        buckets=training_buckets
                    )
                else:
                    # Hunyuan doesn't support full finetune in our UI yet
                    error_msg = "Full finetune is not supported for Hunyuan Video due to memory limitations"
                    logger.error(error_msg)
                    return error_msg, "Training configuration error"
            elif model_type == "ltx_video":
                if training_type == "lora":
                    config = TrainingConfig.ltx_video_lora(
                        data_path=str(TRAINING_PATH),
                        output_path=str(OUTPUT_PATH),
                        buckets=training_buckets
                    )
                else:
                    config = TrainingConfig.ltx_video_full_finetune(
                        data_path=str(TRAINING_PATH),
                        output_path=str(OUTPUT_PATH),
                        buckets=training_buckets
                    )
            elif model_type == "wan":
                if training_type == "lora":
                    config = TrainingConfig.wan_lora(
                        data_path=str(TRAINING_PATH),
                        output_path=str(OUTPUT_PATH),
                        buckets=training_buckets
                    )
                else:
                    error_msg = "Full finetune for Wan is not yet supported in this UI"
                    logger.error(error_msg)
                    return error_msg, "Training configuration error"
            else:
                error_msg = f"Unsupported model type: {model_type}"
                logger.error(error_msg)
                return error_msg, "Unsupported model"
            
            # Update with UI parameters
            config.train_epochs = int(num_epochs)
            config.batch_size = int(batch_size)
            config.lr = float(learning_rate)
            config.checkpointing_steps = int(save_iterations)
            config.training_type = training_type
            config.flow_weighting_scheme = flow_weighting_scheme
            
            # CRITICAL FIX: Update the dataset_config to point to the JSON file, not the directory
            config.data_root = str(dataset_config_file)
            
            # Update LoRA parameters if using LoRA training type
            if training_type == "lora":
                config.lora_rank = int(lora_rank)
                config.lora_alpha = int(lora_alpha)

            # Update with resume_from_checkpoint if provided
            if resume_from_checkpoint:
                config.resume_from_checkpoint = resume_from_checkpoint
                self.append_log(f"Resuming from checkpoint: {resume_from_checkpoint}")
                
            # Common settings for both models
            config.mixed_precision = "bf16"
            config.seed = 42
            config.gradient_checkpointing = True
            config.enable_slicing = True
            config.enable_tiling = True
            config.caption_dropout_p = 0.05

            validation_error = self.validate_training_config(config, model_type)
            if validation_error:
                error_msg = f"Configuration validation failed: {validation_error}"
                logger.error(error_msg)
                return "Error: Invalid configuration", error_msg

            # Convert config to command line arguments for all launcher types
            config_args = config.to_args_list()
            logger.debug("Generated args list: %s", config_args)
            
            # Use different launch commands based on model type
            # For Wan models, use torchrun instead of accelerate launch
            if model_type == "wan":
                # Configure torchrun parameters
                torchrun_args = [
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node=1",
                    "--nnodes=1",
                    "--rdzv_backend=c10d",
                    "--rdzv_endpoint=localhost:0",
                    str(train_script)
                ]
                
                # Additional args needed for torchrun
                config_args.extend([
                    "--parallel_backend", "ptd",
                    "--pp_degree", "1", 
                    "--dp_degree", "1", 
                    "--dp_shards", "1", 
                    "--cp_degree", "1", 
                    "--tp_degree", "1"
                ])
                
                # Log the full command for debugging
                command_str = ' '.join(torchrun_args + config_args)
                self.append_log(f"Command: {command_str}")
                logger.info(f"Executing command: {command_str}")
                
                launch_args = torchrun_args
            else:
                # For other models, use accelerate launch as before
                # Configure accelerate parameters
                accelerate_args = [
                    "accelerate", "launch",
                    "--mixed_precision=bf16",
                    "--num_processes=1",
                    "--num_machines=1",
                    "--dynamo_backend=no",
                    str(train_script)
                ]
                
                # Log the full command for debugging
                command_str = ' '.join(accelerate_args + config_args)
                self.append_log(f"Command: {command_str}")
                logger.info(f"Executing command: {command_str}")
                
                launch_args = accelerate_args
            
            # Set environment variables
            env = os.environ.copy()
            env["NCCL_P2P_DISABLE"] = "1"
            env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
            env["WANDB_MODE"] = "offline"
            env["HF_API_TOKEN"] = HF_API_TOKEN
            env["FINETRAINERS_LOG_LEVEL"] = "DEBUG"  # Added for better debugging
            
            # Start the training process
            process = subprocess.Popen(
                launch_args + config_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=env,
                cwd=str(current_dir),
                bufsize=1,
                universal_newlines=True
            )
            
            logger.info(f"Started process with PID: {process.pid}")
            
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            
            # Save session info including repo_id for later hub upload
            self.save_session({
                "model_type": model_type,
                "training_type": training_type,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "save_iterations": save_iterations,
                "repo_id": repo_id,
                "start_time": datetime.now().isoformat()
            })
            
            # Update initial training status
            total_steps = num_epochs * (max(1, video_count) // batch_size)
            self.save_status(
                state='training',
                epoch=0,
                step=0,
                total_steps=total_steps,
                loss=0.0,
                total_epochs=num_epochs,
                message='Training started',
                repo_id=repo_id,
                model_type=model_type,
                training_type=training_type
            )
            
            # Start monitoring process output
            self._start_log_monitor(process)
            
            success_msg = f"Started {training_type} training for {model_type} model"
            self.append_log(success_msg)
            logger.info(success_msg)
            
            return success_msg, self.get_logs()
            
        except Exception as e:
            error_msg = f"Error {'resuming' if is_resuming else 'starting'} training: {str(e)}"
            self.append_log(error_msg)
            logger.exception("Training startup failed")
            traceback.print_exc()
            return f"Error {'resuming' if is_resuming else 'starting'} training", error_msg
            
    def stop_training(self) -> Tuple[str, str]:
        """Stop training process"""
        if not self.pid_file.exists():
            return "No training process found", self.get_logs()
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                    
            if psutil.pid_exists(pid):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                    
            if self.pid_file.exists():
                self.pid_file.unlink()
                    
            self.append_log("Training process stopped")
            self.save_status(state='stopped', message='Training stopped')
                
            return "Training stopped successfully", self.get_logs()
                
        except Exception as e:
            error_msg = f"Error stopping training: {str(e)}"
            self.append_log(error_msg)
            if self.pid_file.exists():
                self.pid_file.unlink()
            return "Error stopping training", error_msg

    def pause_training(self) -> Tuple[str, str]:
        """Pause training process by sending SIGUSR1"""
        if not self.is_training_running():
            return "No training process found", self.get_logs()
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            if psutil.pid_exists(pid):
                os.kill(pid, signal.SIGUSR1)  # Signal to pause
                self.save_status(state='paused', message='Training paused')
                self.append_log("Training paused")
                
            return "Training paused", self.get_logs()

        except Exception as e:
            error_msg = f"Error pausing training: {str(e)}"
            self.append_log(error_msg)
            return "Error pausing training", error_msg

    def resume_training(self) -> Tuple[str, str]:
        """Resume training process by sending SIGUSR2"""
        if not self.is_training_running():
            return "No training process found", self.get_logs()
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            if psutil.pid_exists(pid):
                os.kill(pid, signal.SIGUSR2)  # Signal to resume
                self.save_status(state='training', message='Training resumed')
                self.append_log("Training resumed")
                
            return "Training resumed", self.get_logs()

        except Exception as e:
            error_msg = f"Error resuming training: {str(e)}"
            self.append_log(error_msg)
            return "Error resuming training", error_msg

    def is_training_running(self) -> bool:
        """Check if training is currently running"""
        if not self.pid_file.exists():
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists AND is a Python process running train.py
            if psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    cmdline = process.cmdline()
                    # Check if it's a Python process running train.py
                    return any('train.py' in cmd for cmd in cmdline)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return False
            return False
        except:
            return False

    def recover_interrupted_training(self) -> Dict[str, Any]:
        """Attempt to recover interrupted training
        
        Returns:
            Dict with recovery status and UI updates
        """
        status = self.get_status()
        ui_updates = {}
        
        # Check for any checkpoints, even if status doesn't indicate training
        checkpoints = list(OUTPUT_PATH.glob("checkpoint-*"))
        has_checkpoints = len(checkpoints) > 0
        
        # If status indicates training but process isn't running, or if we have checkpoints
        # and no active training process, try to recover
        if (status.get('status') in ['training', 'paused'] and not self.is_training_running()) or \
        (has_checkpoints and not self.is_training_running()):
            
            logger.info("Detected interrupted training session or existing checkpoints, attempting to recover...")
            
            # Get the latest checkpoint
            last_session = self.load_session()
            
            if not last_session:
                logger.warning("No session data found for recovery, but will check for checkpoints")
                # Try to create a default session based on UI state if we have checkpoints
                if has_checkpoints:
                    ui_state = self.load_ui_state()
                    # Create a default session using UI state values
                    last_session = {
                        "params": {
                            "model_type": MODEL_TYPES.get(ui_state.get("model_type", list(MODEL_TYPES.keys())[0])),
                            "training_type": TRAINING_TYPES.get(ui_state.get("training_type", list(TRAINING_TYPES.keys())[0])),
                            "lora_rank": ui_state.get("lora_rank", "128"),
                            "lora_alpha": ui_state.get("lora_alpha", "128"),
                            "num_epochs": ui_state.get("num_epochs", 70),
                            "batch_size": ui_state.get("batch_size", 1),
                            "learning_rate": ui_state.get("learning_rate", 3e-5),
                            "save_iterations": ui_state.get("save_iterations", 500),
                            "preset_name": ui_state.get("training_preset", list(TRAINING_PRESETS.keys())[0]),
                            "repo_id": ""  # Default empty repo ID
                        }
                    }
                    logger.info("Created default session from UI state for recovery")
                else:
                    # Set buttons for no active training
                    ui_updates = {
                        "start_btn": {"interactive": True, "variant": "primary", "value": "Start Training"},
                        "stop_btn": {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"},
                        "delete_checkpoints_btn": {"interactive": False, "variant": "stop", "value": "Delete All Checkpoints"},
                        "pause_resume_btn": {"interactive": False, "variant": "secondary", "visible": False}
                    }
                    return {"status": "idle", "message": "No training in progress", "ui_updates": ui_updates}
                
            # Find the latest checkpoint if we have checkpoints
            latest_checkpoint = None
            checkpoint_step = 0
            
            if has_checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getmtime)
                checkpoint_step = int(latest_checkpoint.name.split("-")[1])
                logger.info(f"Found checkpoint at step {checkpoint_step}")
            else:
                logger.warning("No checkpoints found for recovery")
                # Set buttons for no active training
                ui_updates = {
                    "start_btn": {"interactive": True, "variant": "primary", "value": "Start Training"},
                    "stop_btn": {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"},
                    "delete_checkpoints_btn": {"interactive": False, "variant": "stop", "value": "Delete All Checkpoints"},
                    "pause_resume_btn": {"interactive": False, "variant": "secondary", "visible": False}
                }
                return {"status": "error", "message": "No checkpoints found", "ui_updates": ui_updates}
            
            # Extract parameters from the saved session (not current UI state)
            # This ensures we use the original training parameters
            params = last_session.get('params', {})
            
            # Map internal model type back to display name for UI
            model_type_internal = params.get('model_type')
            model_type_display = model_type_internal
            
            # Find the display name that maps to our internal model type
            for display_name, internal_name in MODEL_TYPES.items():
                if internal_name == model_type_internal:
                    model_type_display = display_name
                    logger.info(f"Mapped internal model type '{model_type_internal}' to display name '{model_type_display}'")
                    break
            
            # Get training type (default to LoRA if not present in saved session)
            training_type_internal = params.get('training_type', 'lora')
            training_type_display = next((disp for disp, val in TRAINING_TYPES.items() if val == training_type_internal), list(TRAINING_TYPES.keys())[0])
            
            # Add UI updates to restore the training parameters in the UI
            # This shows the user what values are being used for the resumed training
            ui_updates.update({
                "model_type": model_type_display,  # Use the display name for the UI dropdown
                "training_type": training_type_display,  # Use the display name for training type
                "lora_rank": params.get('lora_rank', "128"),
                "lora_alpha": params.get('lora_alpha', "128"),
                "num_epochs": params.get('num_epochs', 70),
                "batch_size": params.get('batch_size', 1),
                "learning_rate": params.get('learning_rate', 3e-5),
                "save_iterations": params.get('save_iterations', 500),
                "training_preset": params.get('preset_name', list(TRAINING_PRESETS.keys())[0])
            })
            
            # Check if we should auto-recover (immediate restart)
            auto_recover = True  # Always auto-recover on startup
            
            if auto_recover:
                # Rest of the auto-recovery code remains unchanged
                try:
                    # Use the internal model_type for the actual training
                    # But keep model_type_display for the UI
                    result = self.start_training(
                        model_type=model_type_internal,
                        lora_rank=params.get('lora_rank', "128"),
                        lora_alpha=params.get('lora_alpha', "128"),
                        num_epochs=params.get('num_epochs', 70),
                        batch_size=params.get('batch_size', 1),
                        learning_rate=params.get('learning_rate', 3e-5),
                        save_iterations=params.get('save_iterations', 500),
                        repo_id=params.get('repo_id', ''),
                        preset_name=params.get('preset_name', list(TRAINING_PRESETS.keys())[0]),
                        training_type=training_type_internal,
                        resume_from_checkpoint=str(latest_checkpoint)
                    )
                    
                    # Set buttons for active training
                    ui_updates.update({
                        "start_btn": {"interactive": False, "variant": "secondary", "value": "Continue Training"},
                        "stop_btn": {"interactive": True, "variant": "primary", "value": "Stop at Last Checkpoint"},
                        "delete_checkpoints_btn": {"interactive": False, "variant": "stop", "value": "Delete All Checkpoints"},
                        "pause_resume_btn": {"interactive": False, "variant": "secondary", "visible": False}
                    })
                    
                    return {
                        "status": "recovered", 
                        "message": f"Training resumed from checkpoint {checkpoint_step}",
                        "result": result,
                        "ui_updates": ui_updates
                    }
                except Exception as e:
                    logger.error(f"Failed to auto-resume training: {str(e)}")
                    # Set buttons for manual recovery
                    ui_updates.update({
                        "start_btn": {"interactive": True, "variant": "primary", "value": "Continue Training"},
                        "stop_btn": {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"},
                        "delete_checkpoints_btn": {"interactive": True, "variant": "stop", "value": "Delete All Checkpoints"},
                        "pause_resume_btn": {"interactive": False, "variant": "secondary", "visible": False}
                    })
                    return {"status": "error", "message": f"Failed to auto-resume: {str(e)}", "ui_updates": ui_updates}
                else:
                    # Set up UI for manual recovery
                    ui_updates.update({
                        "start_btn": {"interactive": True, "variant": "primary", "value": "Continue Training"},
                        "stop_btn": {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"},
                        "pause_resume_btn": {"interactive": False, "variant": "secondary", "visible": False}
                    })
                    return {"status": "ready_to_recover", "message": f"Ready to resume from checkpoint {checkpoint_step}", "ui_updates": ui_updates}
            
        elif self.is_training_running():
            # Process is still running, set buttons accordingly
            ui_updates = {
                "start_btn": {"interactive": False, "variant": "secondary", "value": "Continue Training" if has_checkpoints else "Start Training"},
                "stop_btn": {"interactive": True, "variant": "primary", "value": "Stop at Last Checkpoint"},
                "pause_resume_btn": {"interactive": False, "variant": "secondary", "visible": False},
                "delete_checkpoints_btn": {"interactive": False, "variant": "stop", "value": "Delete All Checkpoints"}
            }
            return {"status": "running", "message": "Training process is running", "ui_updates": ui_updates}
        else:
            # No training process, set buttons to default state
            button_text = "Continue Training" if has_checkpoints else "Start Training"
            ui_updates = {
                "start_btn": {"interactive": True, "variant": "primary", "value": button_text},
                "stop_btn": {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"},
                "pause_resume_btn": {"interactive": False, "variant": "secondary", "visible": False},
                "delete_checkpoints_btn": {"interactive": has_checkpoints, "variant": "stop", "value": "Delete All Checkpoints"}
            }
            return {"status": "idle", "message": "No training in progress", "ui_updates": ui_updates}
            
    def delete_all_checkpoints(self) -> str:
        """Delete all checkpoints in the output directory.
        
        Returns:
            Status message
        """
        if self.is_training_running():
            return "Cannot delete checkpoints while training is running. Stop training first."
            
        try:
            # Find all checkpoint directories
            checkpoints = list(OUTPUT_PATH.glob("checkpoint-*"))
            
            if not checkpoints:
                return "No checkpoints found to delete."
                
            # Delete each checkpoint directory
            for checkpoint in checkpoints:
                if checkpoint.is_dir():
                    shutil.rmtree(checkpoint)
                    
            # Also delete session.json which contains previous training info
            if self.session_file.exists():
                self.session_file.unlink()
                
            # Reset status file to idle
            self.save_status(state='idle', message='No training in progress')
            
            self.append_log(f"Deleted {len(checkpoints)} checkpoint(s)")
            return f"Successfully deleted {len(checkpoints)} checkpoint(s)"
            
        except Exception as e:
            error_msg = f"Error deleting checkpoints: {str(e)}"
            self.append_log(error_msg)
            return error_msg

    def clear_training_data(self) -> str:
        """Clear all training data"""
        if self.is_training_running():
            return gr.Error("Cannot clear data while training is running")
            
        try:
            for file in TRAINING_VIDEOS_PATH.glob("*.*"):
                file.unlink()
            for file in TRAINING_PATH.glob("*.*"):
                file.unlink()
            
            self.append_log("Cleared all training data")
            return "Training data cleared successfully"
            
        except Exception as e:
            error_msg = f"Error clearing training data: {str(e)}"
            self.append_log(error_msg)
            return error_msg
    
    def save_status(self, state: str, **kwargs) -> None:
        """Save current training status"""
        status = {
            'status': state,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        if state == "Training started" or state == "initializing":
            gr.Info("Initializing model and dataset..")
        elif state == "training":
            gr.Info("Training started!")
        elif state == "completed":
            gr.Info("Training completed!")

        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def _start_log_monitor(self, process: subprocess.Popen) -> None:
        """Start monitoring process output for logs"""
        
        def monitor():
            self.append_log("Starting log monitor thread")
            
            def read_stream(stream, is_error=False):
                if stream:
                    output = stream.readline()
                    if output:
                        # Remove decode() since output is already a string due to universal_newlines=True
                        line = output.strip()
                        if is_error:
                            #self.append_log(f"ERROR: {line}")
                            #logger.error(line)
                            #logger.info(line)
                            self.append_log(line)
                        else:
                            self.append_log(line)
                            # Parse metrics only from stdout
                            metrics = parse_training_log(line)
                            if metrics:
                                status = self.get_status()
                                status.update(metrics)
                                self.save_status(**status)
                        return True
                return False

            # Create separate threads to monitor stdout and stderr
            def monitor_stream(stream, is_error=False):
                while process.poll() is None:
                    if not read_stream(stream, is_error):
                        time.sleep(0.1)  # Short sleep to avoid CPU thrashing
            
            # Start threads to monitor each stream
            stdout_thread = threading.Thread(target=monitor_stream, args=(process.stdout, False))
            stderr_thread = threading.Thread(target=monitor_stream, args=(process.stderr, True))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            process.wait()
            
            # Wait for threads to finish reading any remaining output
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            
            # Process any remaining output after process ends
            while read_stream(process.stdout):
                pass
            while read_stream(process.stderr, True):
                pass
                    
            # Process finished
            return_code = process.poll()
            if return_code == 0:
                success_msg = "Training completed successfully"
                self.append_log(success_msg)
                gr.Info(success_msg)
                self.save_status(state='completed', message=success_msg)
                
                # Upload final model if repository was specified
                session = self.load_session()
                if session and session['params'].get('repo_id'):
                    repo_id = session['params']['repo_id']
                    latest_run = max(Path(OUTPUT_PATH).glob('*'), key=os.path.getmtime)
                    if self.upload_to_hub(latest_run, repo_id):
                        self.append_log(f"Model uploaded to {repo_id}")
                    else:
                        self.append_log("Failed to upload model to hub")
            else:
                error_msg = f"Training failed with return code {return_code}"
                self.append_log(error_msg)
                logger.error(error_msg)
                self.save_status(state='error', message=error_msg)
            
            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

    def upload_to_hub(self, model_path: Path, repo_id: str) -> bool:
        """Upload model to Hugging Face Hub
        
        Args:
            model_path: Path to model files
            repo_id: Repository ID (username/model-name)
            
        Returns:
            bool: Whether upload was successful
        """
        try:
            token = os.getenv("HF_API_TOKEN")
            if not token:
                self.append_log("Error: HF_API_TOKEN not set")
                return False
                
            # Create or get repo
            create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
            
            # Upload files
            upload_folder(
                folder_path=str(OUTPUT_PATH),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Training completed"
            )
            
            return True
        except Exception as e:
            self.append_log(f"Error uploading to hub: {str(e)}")
            return False

    def get_model_output_safetensors(self) -> str:
        """Return the path to the model safetensors
        
            
        Returns:
            Path to created ZIP file
        """
        
        model_output_safetensors_path = OUTPUT_PATH / "pytorch_lora_weights.safetensors"
        return str(model_output_safetensors_path)

    def create_training_dataset_zip(self) -> str:
        """Create a ZIP file containing all training data
        
            
        Returns:
            Path to created ZIP file
        """
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            temp_zip_path = str(temp_zip.name)
            print(f"Creating zip file for {TRAINING_PATH}..")
            try:
                make_archive(TRAINING_PATH, temp_zip_path)
                print(f"Zip file created!")
                return temp_zip_path
            except Exception as e:
                print(f"Failed to create zip: {str(e)}")
                raise gr.Error(f"Failed to create zip: {str(e)}")
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

from .huggingface_hub import upload_folder, create_repo
from .config import TrainingConfig, TRAINING_PRESETS,  LOG_FILE_PATH, TRAINING_VIDEOS_PATH, STORAGE_PATH, TRAINING_PATH, MODEL_PATH, OUTPUT_PATH, HF_API_TOKEN, MODEL_TYPES
from .utils import make_archive, parse_training_log, is_image_file, is_video_file
from .finetrainers_utils import prepare_finetrainers_dataset, copy_files_to_training_dir

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
    
    def save_session(self, params: Dict) -> None:
        """Save training session parameters"""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "status": self.get_status()
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
                #print("status found in the json:", status)
                
            # Check if process is actually running
            if self.pid_file.exists():
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                if not psutil.pid_exists(pid):
                    # Process died unexpectedly
                    if status['status'] == 'training':
                        status['status'] = 'error'
                        status['message'] = 'Training process terminated unexpectedly'
                        self.append_log("Training process terminated unexpectedly")
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
            if not config.data_root or not Path(config.data_root).exists():
                return f"Invalid data root path: {config.data_root}"
                
            if not config.output_dir:
                return "Output directory not specified"
                
            # Check for required files
            videos_file = Path(config.data_root) / "videos.txt"
            prompts_file = Path(config.data_root) / "prompts.txt"
            
            if not videos_file.exists():
                return f"Missing videos list file: {videos_file}"
            if not prompts_file.exists():
                return f"Missing prompts list file: {prompts_file}"
                
            # Validate file counts match
            video_lines = [l.strip() for l in open(videos_file) if l.strip()]
            prompt_lines = [l.strip() for l in open(prompts_file) if l.strip()]
            
            if not video_lines:
                return "No training files found"
            if len(video_lines) != len(prompt_lines):
                return f"Mismatch between video count ({len(video_lines)}) and prompt count ({len(prompt_lines)})"
                
            # Model-specific validation
            if model_type == "hunyuan_video":
                if config.batch_size > 2:
                    return "Hunyuan model recommended batch size is 1-2"
                if not config.gradient_checkpointing:
                    return "Gradient checkpointing is required for Hunyuan model"
            elif model_type == "ltx_video":
                if config.batch_size > 4:
                    return "LTX model recommended batch size is 1-4"
                    
            logger.info(f"Config validation passed with {len(video_lines)} training files")
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
    ) -> Tuple[str, str]:
        """Start training with finetrainers"""
            
        self.clear_logs()

        if not model_type:
            raise ValueError("model_type cannot be empty")
        if model_type not in MODEL_TYPES.values():
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(MODEL_TYPES.values())}")


        logger.info(f"Initializing training with model_type={model_type}")
        
        try:
            # Get absolute paths
            current_dir = Path(__file__).parent.absolute()
            train_script = current_dir / "train.py"
            
            if not train_script.exists():
                error_msg = f"Training script not found at {train_script}"
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

            # Get config for selected model type with preset buckets
            if model_type == "hunyuan_video":
                config = TrainingConfig.hunyuan_video_lora(
                    data_path=str(TRAINING_PATH),
                    output_path=str(OUTPUT_PATH),
                    buckets=training_buckets
                )
            else:  # ltx_video
                config = TrainingConfig.ltx_video_lora(
                    data_path=str(TRAINING_PATH),
                    output_path=str(OUTPUT_PATH),
                    buckets=training_buckets
                )
            
            # Update with UI parameters
            config.train_epochs = int(num_epochs)
            config.lora_rank = int(lora_rank)
            config.lora_alpha = int(lora_alpha)
            config.batch_size = int(batch_size)
            config.lr = float(learning_rate)
            config.checkpointing_steps = int(save_iterations)

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

            # Configure accelerate parameters
            accelerate_args = [
                "accelerate", "launch",
                "--mixed_precision=bf16",
                "--num_processes=1",
                "--num_machines=1",
                "--dynamo_backend=no"
            ]
            
            accelerate_args.append(str(train_script))
            
            # Convert config to command line arguments
            config_args = config.to_args_list()
            

            logger.debug("Generated args list: %s", config_args)

            # Log the full command for debugging
            command_str = ' '.join(accelerate_args + config_args)
            self.append_log(f"Command: {command_str}")
            logger.info(f"Executing command: {command_str}")
            
            # Set environment variables
            env = os.environ.copy()
            env["NCCL_P2P_DISABLE"] = "1"
            env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
            env["WANDB_MODE"] = "offline"
            env["HF_API_TOKEN"] = HF_API_TOKEN
            env["FINETRAINERS_LOG_LEVEL"] = "DEBUG"  # Added for better debugging
            
            # Start the training process
            process = subprocess.Popen(
                accelerate_args + config_args,
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
                model_type=model_type
            )
            
            # Start monitoring process output
            self._start_log_monitor(process)
            
            success_msg = f"Started training {model_type} model"
            self.append_log(success_msg)
            logger.info(success_msg)
            
            return success_msg, self.get_logs()
            
        except Exception as e:
            error_msg = f"Error starting training: {str(e)}"
            self.append_log(error_msg)
            logger.exception("Training startup failed")
            traceback.print_exc()  # Added for better error debugging
            return "Error starting training", error_msg
        
        
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
            return psutil.pid_exists(pid)
        except:
            return False

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

            # Use select to monitor both stdout and stderr
            while process.poll() is None:
                outputs = [process.stdout, process.stderr]
                readable, _, _ = select.select(outputs, [], [], 1.0)
                
                for stream in readable:
                    is_error = (stream == process.stderr)
                    read_stream(stream, is_error)

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


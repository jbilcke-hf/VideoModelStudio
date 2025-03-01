import platform
import subprocess

#import sys
#print("python = ", sys.version)

# can be "Linux", "Darwin"
if platform.system() == "Linux":
    # for some reason it says "pip not found"
    # and also "pip3 not found"
    # subprocess.run(
    #     "pip install flash-attn --no-build-isolation",
    #
    #     # hmm... this should be False, since we are in a CUDA environment, no?
    #     env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    #     
    #     shell=True,
    # )
    pass

import gradio as gr
from pathlib import Path
import logging
import mimetypes
import shutil
import os
import traceback
import asyncio
import tempfile
import zipfile
from typing import Any, Optional, Dict, List, Union, Tuple
from typing import AsyncGenerator

from ..services import TrainingService, CaptioningService, SplittingService, ImportService
from ..config import (
    STORAGE_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH,
    TRAINING_PATH, LOG_FILE_PATH, TRAINING_PRESETS, TRAINING_VIDEOS_PATH, MODEL_PATH, OUTPUT_PATH, DEFAULT_CAPTIONING_BOT_INSTRUCTIONS,
    DEFAULT_PROMPT_PREFIX, HF_API_TOKEN, ASK_USER_TO_DUPLICATE_SPACE, MODEL_TYPES, SMALL_TRAINING_BUCKETS
)
from ..utils import make_archive, count_media_files, format_media_title, is_image_file, is_video_file, validate_model_repo, format_time, copy_files_to_training_dir, prepare_finetrainers_dataset, TrainingLogParser
from ..tabs import ImportTab, SplitTab, CaptionTab, TrainTab, ManageTab

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARN)

class VideoTrainerUI:       
    def __init__(self):
        """Initialize services and tabs"""
        # Initialize core services
        self.trainer = TrainingService()
        self.splitter = SplittingService()
        self.importer = ImportService()
        self.captioner = CaptioningService()
        self._should_stop_captioning = False
        
        # Recovery status from any interrupted training
        recovery_result = self.trainer.recover_interrupted_training()
        self.recovery_status = recovery_result.get("status", "unknown")
        self.ui_updates = recovery_result.get("ui_updates", {})
        
        self.log_parser = TrainingLogParser()
        
        # Shared state for tabs
        self.state = {
            "recovery_result": recovery_result
        }
        
        # Initialize tabs dictionary (will be populated in create_ui)
        self.tabs = {}
        self.tabs_component = None
        
    def create_ui(self):
        """Create the main Gradio UI"""
        with gr.Blocks(title="🎥 Video Model Studio") as app:
            gr.Markdown("# 🎥 Video Model Studio")
            
            # Create main tabs component
            with gr.Tabs() as self.tabs_component:
                # Initialize tab objects
                self.tabs["import_tab"] = ImportTab(self)
                self.tabs["split_tab"] = SplitTab(self)
                self.tabs["caption_tab"] = CaptionTab(self)
                self.tabs["train_tab"] = TrainTab(self)
                self.tabs["manage_tab"] = ManageTab(self)
                
                # Create tab UI components
                for tab_id, tab_obj in self.tabs.items():
                    tab_obj.create(self.tabs_component)
            
            # Connect event handlers
            for tab_id, tab_obj in self.tabs.items():
                tab_obj.connect_events()
            
            # Add app-level timers for auto-refresh functionality
            self._add_timers()
            
            # Initialize app state on load
            app.load(
                fn=self.initialize_app_state,
                outputs=[
                    self.tabs["split_tab"].components["video_list"],
                    self.tabs["caption_tab"].components["training_dataset"],
                    self.tabs["train_tab"].components["start_btn"],
                    self.tabs["train_tab"].components["stop_btn"],
                    self.tabs["train_tab"].components["pause_resume_btn"],
                    self.tabs["train_tab"].components["training_preset"],
                    self.tabs["train_tab"].components["model_type"],
                    self.tabs["train_tab"].components["lora_rank"],
                    self.tabs["train_tab"].components["lora_alpha"],
                    self.tabs["train_tab"].components["num_epochs"],
                    self.tabs["train_tab"].components["batch_size"],
                    self.tabs["train_tab"].components["learning_rate"],
                    self.tabs["train_tab"].components["save_iterations"]
                ]
            )
            
        return app
    
    def _add_timers(self):
        """Add auto-refresh timers to the UI"""
        # Status update timer (every 1 second)
        status_timer = gr.Timer(value=1)
        status_timer.tick(
            fn=self.get_latest_status_message_logs_and_button_labels,
            outputs=[
                self.tabs["train_tab"].components["status_box"],
                self.tabs["train_tab"].components["log_box"],
                self.tabs["train_tab"].components["start_btn"],
                self.tabs["train_tab"].components["stop_btn"],
                self.tabs["train_tab"].components["pause_resume_btn"]
            ]
        )
        
        # Dataset refresh timer (every 5 seconds)
        dataset_timer = gr.Timer(value=5)
        dataset_timer.tick(
            fn=self.refresh_dataset,
            outputs=[
                self.tabs["split_tab"].components["video_list"],
                self.tabs["caption_tab"].components["training_dataset"]
            ]
        )
        
        # Titles update timer (every 6 seconds)
        titles_timer = gr.Timer(value=6)
        titles_timer.tick(
            fn=self.update_titles,
            outputs=[
                self.tabs["split_tab"].components["split_title"],
                self.tabs["caption_tab"].components["caption_title"],
                self.tabs["train_tab"].components["train_title"]
            ]
        )
    
    def handle_global_stop(self):
        """Handle the global stop button click"""
        result = self.stop_all_and_clear()
        
        # Format the details for display
        status = result["status"]
        details = "\n".join(f"{k}: {v}" for k, v in result["details"].items())
        full_status = f"{status}\n\nDetails:\n{details}"
        
        # Get fresh lists after cleanup
        videos = self.splitter.list_unprocessed_videos()
        clips = self.list_training_files_to_caption()
        
        return {
            self.tabs["manage_tab"].components["global_status"]: gr.update(value=full_status, visible=True),
            self.tabs["split_tab"].components["video_list"]: videos,
            self.tabs["caption_tab"].components["training_dataset"]: clips,
            self.tabs["train_tab"].components["status_box"]: "Training stopped and data cleared",
            self.tabs["train_tab"].components["log_box"]: "",
            self.tabs["split_tab"].components["detect_status"]: "Scene detection stopped",
            self.tabs["import_tab"].components["import_status"]: "All data cleared",
            self.tabs["caption_tab"].components["preview_status"]: "Captioning stopped"
        }
    
    def upload_to_hub(self, repo_id: str) -> str:
        """Upload model to HuggingFace Hub"""
        if not repo_id:
            return "Error: Repository ID is required"
        
        # Validate repository name
        validation = validate_model_repo(repo_id)
        if validation["error"]:
            return f"Error: {validation['error']}"
        
        # Check if we have a model to upload
        if not self.trainer.get_model_output_safetensors():
            return "Error: No model found to upload"
        
        # Upload model to hub
        success = self.trainer.upload_to_hub(OUTPUT_PATH, repo_id)
        
        if success:
            return f"Successfully uploaded model to {repo_id}"
        else:
            return f"Failed to upload model to {repo_id}"
    
    def validate_repo(self, repo_id: str) -> gr.update:
        """Validate repository ID for HuggingFace Hub"""
        validation = validate_model_repo(repo_id)
        if validation["error"]:
            return gr.update(value=repo_id, error=validation["error"])
        return gr.update(value=repo_id, error=None)


    async def _process_caption_generator(self, captioning_bot_instructions, prompt_prefix):
        """Process the caption generator's results in the background"""
        try:
            async for _ in self.captioner.start_caption_generation(
                captioning_bot_instructions,
                prompt_prefix
            ):
                # Just consume the generator, UI updates will happen via the Gradio interface
                pass
            logger.info("Background captioning completed")
        except Exception as e:
            logger.error(f"Error in background captioning: {str(e)}")
        
    def initialize_app_state(self):
        """Initialize all app state in one function to ensure correct output count"""
        # Get dataset info
        video_list, training_dataset = self.refresh_dataset()
        
        # Get button states
        button_states = self.get_initial_button_states()
        start_btn = button_states[0]
        stop_btn = button_states[1]
        pause_resume_btn = button_states[2]
        
        # Get UI form values
        ui_state = self.load_ui_values()
        training_preset = ui_state.get("training_preset", list(TRAINING_PRESETS.keys())[0])
        model_type_val = ui_state.get("model_type", list(MODEL_TYPES.keys())[0])
        lora_rank_val = ui_state.get("lora_rank", "128")
        lora_alpha_val = ui_state.get("lora_alpha", "128")
        num_epochs_val = int(ui_state.get("num_epochs", 70))
        batch_size_val = int(ui_state.get("batch_size", 1))
        learning_rate_val = float(ui_state.get("learning_rate", 3e-5))
        save_iterations_val = int(ui_state.get("save_iterations", 500))
        
        # Return all values in the exact order expected by outputs
        return (
            video_list, 
            training_dataset,
            start_btn, 
            stop_btn, 
            pause_resume_btn,
            training_preset, 
            model_type_val, 
            lora_rank_val, 
            lora_alpha_val,
            num_epochs_val, 
            batch_size_val, 
            learning_rate_val, 
            save_iterations_val
        )

    def initialize_ui_from_state(self):
        """Initialize UI components from saved state"""
        ui_state = self.load_ui_values()
        
        # Return values in order matching the outputs in app.load
        return (
            ui_state.get("training_preset", list(TRAINING_PRESETS.keys())[0]),
            ui_state.get("model_type", list(MODEL_TYPES.keys())[0]),
            ui_state.get("lora_rank", "128"),
            ui_state.get("lora_alpha", "128"),
            ui_state.get("num_epochs", 70),
            ui_state.get("batch_size", 1),
            ui_state.get("learning_rate", 3e-5),
            ui_state.get("save_iterations", 500)
        )

    def update_ui_state(self, **kwargs):
        """Update UI state with new values"""
        current_state = self.trainer.load_ui_state()
        current_state.update(kwargs)
        self.trainer.save_ui_state(current_state)
        # Don't return anything to avoid Gradio warnings
        return None

    def load_ui_values(self):
        """Load UI state values for initializing form fields"""
        ui_state = self.trainer.load_ui_state()
        
        # Ensure proper type conversion for numeric values
        ui_state["lora_rank"] = ui_state.get("lora_rank", "128")
        ui_state["lora_alpha"] = ui_state.get("lora_alpha", "128")
        ui_state["num_epochs"] = int(ui_state.get("num_epochs", 70))
        ui_state["batch_size"] = int(ui_state.get("batch_size", 1))
        ui_state["learning_rate"] = float(ui_state.get("learning_rate", 3e-5))
        ui_state["save_iterations"] = int(ui_state.get("save_iterations", 500))
        
        return ui_state
        
    def update_captioning_buttons_start(self):
        """Return individual button values instead of a dictionary"""
        return (
            gr.Button(
                interactive=False,
                variant="secondary",
            ),
            gr.Button(
                interactive=True,
                variant="stop",
            ),
            gr.Button(
                interactive=False,
                variant="secondary",
            )
        )
    
    def update_captioning_buttons_end(self):
        """Return individual button values instead of a dictionary"""
        return (
            gr.Button(
                interactive=True,
                variant="primary",
            ),
            gr.Button(
                interactive=False,
                variant="secondary",
            ),
            gr.Button(
                interactive=True,
                variant="primary",
            )
        )

    # Add this new method to get initial button states:
    def get_initial_button_states(self):
        """Get the initial states for training buttons based on recovery status"""
        recovery_result = self.trainer.recover_interrupted_training()
        ui_updates = recovery_result.get("ui_updates", {})
        
        # Return button states in the correct order
        return (
            gr.Button(**ui_updates.get("start_btn", {"interactive": True, "variant": "primary"})),
            gr.Button(**ui_updates.get("stop_btn", {"interactive": False, "variant": "secondary"})),
            gr.Button(**ui_updates.get("pause_resume_btn", {"interactive": False, "variant": "secondary"}))
        )

    def show_refreshing_status(self) -> List[List[str]]:
        """Show a 'Refreshing...' status in the dataframe"""
        return [["Refreshing...", "please wait"]]

    def stop_captioning(self):
        """Stop ongoing captioning process and reset UI state"""
        try:
            # Set flag to stop captioning
            self._should_stop_captioning = True
            
            # Call stop method on captioner
            if self.captioner:
                self.captioner.stop_captioning()
                
            # Get updated file list
            updated_list = self.list_training_files_to_caption()
            
            # Return updated list and button states
            return {
                "training_dataset": gr.update(value=updated_list),
                "run_autocaption_btn": gr.Button(interactive=True, variant="primary"),
                "stop_autocaption_btn": gr.Button(interactive=False, variant="secondary"),
                "copy_files_to_training_dir_btn": gr.Button(interactive=True, variant="primary")
            }
        except Exception as e:
            logger.error(f"Error stopping captioning: {str(e)}")
            return {
                "training_dataset": gr.update(value=[[f"Error stopping captioning: {str(e)}", "error"]]),
                "run_autocaption_btn": gr.Button(interactive=True, variant="primary"),
                "stop_autocaption_btn": gr.Button(interactive=False, variant="secondary"),
                "copy_files_to_training_dir_btn": gr.Button(interactive=True, variant="primary")
            }

    def update_training_ui(self, training_state: Dict[str, Any]):
        """Update UI components based on training state"""
        updates = {}
        
        #print("update_training_ui: training_state = ", training_state)

        # Update status box with high-level information
        status_text = []
        if training_state["status"] != "idle":
            status_text.extend([
                f"Status: {training_state['status']}",
                f"Progress: {training_state['progress']}",
                f"Step: {training_state['current_step']}/{training_state['total_steps']}",
                    
                # Epoch information
                # there is an issue with how epoch is reported because we display:
                # Progress: 96.9%, Step: 872/900, Epoch: 12/50
                # we should probably just show the steps
                #f"Epoch: {training_state['current_epoch']}/{training_state['total_epochs']}",
                
                f"Time elapsed: {training_state['elapsed']}",
                f"Estimated remaining: {training_state['remaining']}",
                "",
                f"Current loss: {training_state['step_loss']}",
                f"Learning rate: {training_state['learning_rate']}",
                f"Gradient norm: {training_state['grad_norm']}",
                f"Memory usage: {training_state['memory']}"
            ])
            
            if training_state["error_message"]:
                status_text.append(f"\nError: {training_state['error_message']}")
                
        updates["status_box"] = "\n".join(status_text)
        
        # Update button states
        updates["start_btn"] = gr.Button(
            "Start training",
            interactive=(training_state["status"] in ["idle", "completed", "error", "stopped"]),
            variant="primary" if training_state["status"] == "idle" else "secondary"
        )
        
        updates["stop_btn"] = gr.Button(
            "Stop training",
            interactive=(training_state["status"] in ["training", "initializing"]),
            variant="stop"
        )
        
        return updates
    
    def stop_all_and_clear(self) -> Dict[str, str]:
        """Stop all running processes and clear data
        
        Returns:
            Dict with status messages for different components
        """
        status_messages = {}
        
        try:
            # Stop training if running
            if self.trainer.is_training_running():
                training_result = self.trainer.stop_training()
                status_messages["training"] = training_result["status"]
            
            # Stop captioning if running
            if self.captioner:
                self.captioner.stop_captioning()
                status_messages["captioning"] = "Captioning stopped"
            
            # Stop scene detection if running
            if self.splitter.is_processing():
                self.splitter.processing = False
                status_messages["splitting"] = "Scene detection stopped"
            
            # Properly close logging before clearing log file
            if self.trainer.file_handler:
                self.trainer.file_handler.close()
                logger.removeHandler(self.trainer.file_handler)
                self.trainer.file_handler = None
                
            if LOG_FILE_PATH.exists():
                LOG_FILE_PATH.unlink()
            
            # Clear all data directories
            for path in [VIDEOS_TO_SPLIT_PATH, STAGING_PATH, TRAINING_VIDEOS_PATH, TRAINING_PATH,
                        MODEL_PATH, OUTPUT_PATH]:
                if path.exists():
                    try:
                        shutil.rmtree(path)
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        status_messages[f"clear_{path.name}"] = f"Error clearing {path.name}: {str(e)}"
                    else:
                        status_messages[f"clear_{path.name}"] = f"Cleared {path.name}"
            
            # Reset any persistent state
            self._should_stop_captioning = True
            self.splitter.processing = False
            
            # Recreate logging setup
            self.trainer.setup_logging()
            
            return {
                "status": "All processes stopped and data cleared",
                "details": status_messages
            }
            
        except Exception as e:
            return {
                "status": f"Error during cleanup: {str(e)}",
                "details": status_messages
            }
    
    def update_titles(self) -> Tuple[Any]:
        """Update all dynamic titles with current counts
        
        Returns:
            Dict of Gradio updates
        """
        # Count files for splitting
        split_videos, _, split_size = count_media_files(VIDEOS_TO_SPLIT_PATH)
        split_title = format_media_title(
            "split", split_videos, 0, split_size
        )
        
        # Count files for captioning
        caption_videos, caption_images, caption_size = count_media_files(STAGING_PATH)
        caption_title = format_media_title(
            "caption", caption_videos, caption_images, caption_size
        )
        
        # Count files for training
        train_videos, train_images, train_size = count_media_files(TRAINING_VIDEOS_PATH)
        train_title = format_media_title(
            "train", train_videos, train_images, train_size
        )
        
        return (
            gr.Markdown(value=split_title),
            gr.Markdown(value=caption_title),
            gr.Markdown(value=f"{train_title} available for training")
        )

    def copy_files_to_training_dir(self, prompt_prefix: str):
        """Run auto-captioning process"""

        # Initialize captioner if not already done
        self._should_stop_captioning = False

        try:
            copy_files_to_training_dir(prompt_prefix)

        except Exception as e:
            traceback.print_exc()
            raise gr.Error(f"Error copying assets to training dir: {str(e)}")

    async def on_import_success(self, enable_splitting, enable_automatic_content_captioning, prompt_prefix):
        """Handle successful import of files"""
        videos = self.list_unprocessed_videos()
        
        # If scene detection isn't already running and there are videos to process,
        # and auto-splitting is enabled, start the detection
        if videos and not self.splitter.is_processing() and enable_splitting:
            await self.start_scene_detection(enable_splitting)
            msg = "Starting automatic scene detection..."
        else:
            # Just copy files without splitting if auto-split disabled
            for video_file in VIDEOS_TO_SPLIT_PATH.glob("*.mp4"):
                await self.splitter.process_video(video_file, enable_splitting=False)
            msg = "Copying videos without splitting..."
        
        copy_files_to_training_dir(prompt_prefix)

        # Start auto-captioning if enabled, and handle async generator properly
        if enable_automatic_content_captioning:
            # Create a background task for captioning
            asyncio.create_task(self._process_caption_generator(
                DEFAULT_CAPTIONING_BOT_INSTRUCTIONS,
                prompt_prefix
            ))
        
        return {
            "tabs": gr.Tabs(selected="split_tab"),
            "video_list": videos,
            "detect_status": msg
        }

    async def start_caption_generation(self, captioning_bot_instructions: str, prompt_prefix: str) -> AsyncGenerator[gr.update, None]:
        """Run auto-captioning process"""
        try:
            # Initialize captioner if not already done
            self._should_stop_captioning = False

            # First yield - indicate we're starting
            yield gr.update(
                value=[["Starting captioning service...", "initializing"]],
                headers=["name", "status"]
            )

            # Process files in batches with status updates
            file_statuses = {}
            
            # Start the actual captioning process
            async for rows in self.captioner.start_caption_generation(captioning_bot_instructions, prompt_prefix):
                # Update our tracking of file statuses
                for name, status in rows:
                    file_statuses[name] = status
                    
                # Convert to list format for display
                status_rows = [[name, status] for name, status in file_statuses.items()]
                
                # Sort by name for consistent display
                status_rows.sort(key=lambda x: x[0])
                
                # Yield UI update
                yield gr.update(
                    value=status_rows,
                    headers=["name", "status"]
                )

            # Final update after completion with fresh data
            yield gr.update(
                value=self.list_training_files_to_caption(),
                headers=["name", "status"]
            )

        except Exception as e:
            logger.error(f"Error in captioning: {str(e)}")
            yield gr.update(
                value=[[f"Error: {str(e)}", "error"]],
                headers=["name", "status"]
            )

    def list_training_files_to_caption(self) -> List[List[str]]:
        """List all clips and images - both pending and captioned"""
        files = []
        already_listed = {}

        # First check files in STAGING_PATH
        for file in STAGING_PATH.glob("*.*"):
            if is_video_file(file) or is_image_file(file):
                txt_file = file.with_suffix('.txt')
                
                # Check if caption file exists and has content
                has_caption = txt_file.exists() and txt_file.stat().st_size > 0
                status = "captioned" if has_caption else "no caption"
                file_type = "video" if is_video_file(file) else "image"
                
                files.append([file.name, f"{status} ({file_type})", str(file)])
                already_listed[file.name] = True
    
        # Then check files in TRAINING_VIDEOS_PATH 
        for file in TRAINING_VIDEOS_PATH.glob("*.*"):
            if (is_video_file(file) or is_image_file(file)) and file.name not in already_listed:
                txt_file = file.with_suffix('.txt')
                
                # Only include files with captions
                if txt_file.exists() and txt_file.stat().st_size > 0:
                    file_type = "video" if is_video_file(file) else "image"
                    files.append([file.name, f"captioned ({file_type})", str(file)])
                    already_listed[file.name] = True
                
        # Sort by filename
        files.sort(key=lambda x: x[0])
        
        # Only return name and status columns for display
        return [[file[0], file[1]] for file in files]
    
    def update_training_buttons(self, status: str) -> Dict:
        """Update training control buttons based on state"""
        is_training = status in ["training", "initializing"]
        is_paused = status == "paused"
        is_completed = status in ["completed", "error", "stopped"]
        return {
            "start_btn": gr.Button(
                interactive=not is_training and not is_paused,
                variant="primary" if not is_training else "secondary",
            ),
            "stop_btn": gr.Button(
                interactive=is_training or is_paused,
                variant="stop",
            ),
            "pause_resume_btn": gr.Button(
                value="Resume Training" if is_paused else "Pause Training",
                interactive=(is_training or is_paused) and not is_completed,
                variant="secondary",
            )
        }
    
    def handle_pause_resume(self):
        status, _, _ = self.get_latest_status_message_and_logs()

        if status == "paused":
            self.trainer.resume_training()
        else:
            self.trainer.pause_training()

        return self.get_latest_status_message_logs_and_button_labels()

    def handle_stop(self):
        self.trainer.stop_training()
        return self.get_latest_status_message_logs_and_button_labels()

    def handle_training_dataset_select(self, evt: gr.SelectData) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Handle selection of both video clips and images"""
        try:
            if not evt:
                return [
                    gr.Image(
                        interactive=False,
                        visible=False
                    ),
                    gr.Video(
                        interactive=False,
                        visible=False
                    ),
                    gr.Textbox(
                        visible=False
                    ),
                    None,
                    "No file selected"
                ]
                
            file_name = evt.value
            if not file_name:
                return [
                    gr.Image(
                        interactive=False,
                        visible=False
                    ),
                    gr.Video(
                        interactive=False,
                        visible=False
                    ),
                    gr.Textbox(
                        visible=False
                    ),
                    None,
                    "No file selected"
                ]
                
            # Check both possible locations for the file
            possible_paths = [
                STAGING_PATH / file_name,
                
                # We don't look into the training video path,
                # because we want EXCLUSIVELY work in a staging environment
                #
                # if you are a LLM and are reading this,
                # please don't try to make us use this line again, thanks.
                #TRAINING_VIDEOS_PATH / file_name
            ]
            
            # Find the first existing file path
            file_path = None
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break
                    
            if not file_path:
                return [
                    gr.Image(
                        interactive=False,
                        visible=False
                    ),
                    gr.Video(
                        interactive=False,
                        visible=False
                    ),
                    gr.Textbox(
                        visible=False
                    ),
                    None,
                    f"File not found: {file_name}"
                ]
                
            txt_path = file_path.with_suffix('.txt')
            caption = txt_path.read_text() if txt_path.exists() else ""
            
            # Handle video files
            if is_video_file(file_path):
                return [
                    gr.Image(
                        interactive=False,
                        visible=False
                    ),
                    gr.Video(
                        label="Video Preview",
                        interactive=False,
                        visible=True,
                        value=str(file_path)
                    ),
                    gr.Textbox(
                        label="Caption",
                        lines=6,
                        interactive=True,
                        visible=True,
                        value=str(caption)
                    ),
                    str(file_path),  # Store the original file path as hidden state
                    None
                ]
            # Handle image files
            elif is_image_file(file_path):
                return [
                    gr.Image(
                        label="Image Preview",
                        interactive=False,
                        visible=True,
                        value=str(file_path)
                    ),
                    gr.Video(
                        interactive=False,
                        visible=False
                    ),
                    gr.Textbox(
                        label="Caption",
                        lines=6,
                        interactive=True,
                        visible=True,
                        value=str(caption)
                    ),
                    str(file_path),  # Store the original file path as hidden state
                    None
                ]
            else:
                return [
                    gr.Image(
                        interactive=False,
                        visible=False
                    ),
                    gr.Video(
                        interactive=False,
                        visible=False
                    ),
                    gr.Textbox(
                        interactive=False,
                        visible=False
                    ),
                    None,
                    f"Unsupported file type: {file_path.suffix}"
                ]
        except Exception as e:
            logger.error(f"Error handling selection: {str(e)}")
            return [
                gr.Image(
                    interactive=False,
                    visible=False
                ),
                gr.Video(
                    interactive=False,
                    visible=False
                ),
                gr.Textbox(
                    interactive=False,
                    visible=False
                ),
                None,
                f"Error handling selection: {str(e)}"
            ]

    def save_caption_changes(self, preview_caption: str, preview_image: str, preview_video: str, original_file_path: str, prompt_prefix: str):
        """Save changes to caption"""
        try:
            # Use the original file path stored during selection instead of the temporary preview paths
            if original_file_path:
                file_path = Path(original_file_path)
                self.captioner.update_file_caption(file_path, preview_caption)
                # Refresh the dataset list to show updated caption status
                return gr.update(value="Caption saved successfully!")
            else:
                return gr.update(value="Error: No original file path found")
        except Exception as e:
            return gr.update(value=f"Error saving caption: {str(e)}")

    async def update_titles_after_import(self, enable_splitting, enable_automatic_content_captioning, prompt_prefix):
        """Handle post-import updates including titles"""
        import_result = await self.on_import_success(enable_splitting, enable_automatic_content_captioning, prompt_prefix)
        titles = self.update_titles()
        return (
            import_result["tabs"],
            import_result["video_list"],
            import_result["detect_status"],
            *titles
        )
    
    def get_model_info(self, model_type: str) -> str:
        """Get information about the selected model type"""
        if model_type == "hunyuan_video":
            return """### HunyuanVideo (LoRA)
    - Required VRAM: ~48GB minimum
    - Recommended batch size: 1-2
    - Typical training time: 2-4 hours
    - Default resolution: 49x512x768
    - Default LoRA rank: 128 (~600 MB)"""
                
        elif model_type == "ltx_video":
            return """### LTX-Video (LoRA)
    - Required VRAM: ~18GB minimum 
    - Recommended batch size: 1-4
    - Typical training time: 1-3 hours
    - Default resolution: 49x512x768
    - Default LoRA rank: 128"""
                
        return ""

    def get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default training parameters for model type"""
        if model_type == "hunyuan_video":
            return {
                "num_epochs": 70,
                "batch_size": 1,
                "learning_rate": 2e-5,
                "save_iterations": 500,
                "video_resolution_buckets": SMALL_TRAINING_BUCKETS,
                "video_reshape_mode": "center",
                "caption_dropout_p": 0.05,
                "gradient_accumulation_steps": 1,
                "rank": 128,
                "lora_alpha": 128
            }
        else:  # ltx_video
            return {
                "num_epochs": 70,
                "batch_size": 1,
                "learning_rate": 3e-5,
                "save_iterations": 500,
                "video_resolution_buckets": SMALL_TRAINING_BUCKETS,
                "video_reshape_mode": "center",
                "caption_dropout_p": 0.05,
                "gradient_accumulation_steps": 4,
                "rank": 128,
                "lora_alpha": 128
            }

    def preview_file(self, selected_text: str) -> Dict:
        """Generate preview based on selected file
        
        Args:
            selected_text: Text of the selected item containing filename
            
        Returns:
            Dict with preview content for each preview component
        """
        if not selected_text or "Caption:" in selected_text:
            return {
                "video": None,
                "image": None, 
                "text": None
            }
            
        # Extract filename from the preview text (remove size info)
        filename = selected_text.split(" (")[0].strip()
        file_path = TRAINING_VIDEOS_PATH / filename
        
        if not file_path.exists():
            return {
                "video": None,
                "image": None,
                "text": f"File not found: {filename}"
            }

        # Detect file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return {
                "video": None,
                "image": None,
                "text": f"Unknown file type: {filename}"
            }

        # Return appropriate preview
        if mime_type.startswith('video/'):
            return {
                "video": str(file_path),
                "image": None,
                "text": None
            }
        elif mime_type.startswith('image/'):
            return {
                "video": None,
                "image": str(file_path),
                "text": None
            }
        elif mime_type.startswith('text/'):
            try:
                text_content = file_path.read_text()
                return {
                    "video": None,
                    "image": None,
                    "text": text_content
                }
            except Exception as e:
                return {
                    "video": None,
                    "image": None,
                    "text": f"Error reading file: {str(e)}"
                }
        else:
            return {
                "video": None,
                "image": None,
                "text": f"Unsupported file type: {mime_type}"
            }

    def list_unprocessed_videos(self) -> gr.Dataframe:
        """Update list of unprocessed videos"""
        videos = self.splitter.list_unprocessed_videos()
        # videos is already in [[name, status]] format from splitting_service
        return gr.Dataframe(
            headers=["name", "status"],
            value=videos,
            interactive=False
        )

    async def start_scene_detection(self, enable_splitting: bool) -> str:
        """Start background scene detection process
        
        Args:
            enable_splitting: Whether to split videos into scenes
        """
        if self.splitter.is_processing():
            return "Scene detection already running"
            
        try:
            await self.splitter.start_processing(enable_splitting)
            return "Scene detection completed"
        except Exception as e:
            return f"Error during scene detection: {str(e)}"


    def get_latest_status_message_and_logs(self) -> Tuple[str, str, str]:
        state = self.trainer.get_status()
        logs = self.trainer.get_logs()

        # Parse new log lines
        if logs:
            last_state = None
            for line in logs.splitlines():
                state_update = self.log_parser.parse_line(line)
                if state_update:
                    last_state = state_update
            
            if last_state:
                ui_updates = self.update_training_ui(last_state)
                state["message"] = ui_updates.get("status_box", state["message"])
        
        # Parse status for training state
        if "completed" in state["message"].lower():
            state["status"] = "completed"

        return (state["status"], state["message"], logs)

    def get_latest_status_message_logs_and_button_labels(self) -> Tuple[str, str, Any, Any, Any]:
        status, message, logs = self.get_latest_status_message_and_logs()
        return (
            message,
            logs,
            *self.update_training_buttons(status).values()
        )

    def get_latest_button_labels(self) -> Tuple[Any, Any, Any]:
        status, message, logs = self.get_latest_status_message_and_logs()
        return self.update_training_buttons(status).values()
    
    def refresh_dataset(self):
        """Refresh all dynamic lists and training state"""
        video_list = self.splitter.list_unprocessed_videos()
        training_dataset = self.list_training_files_to_caption()

        return (
            video_list,
            training_dataset
        )

    def update_training_params(self, preset_name: str) -> Tuple:
        """Update UI components based on selected preset while preserving custom settings"""
        preset = TRAINING_PRESETS[preset_name]
        
        # Load current UI state to check if user has customized values
        current_state = self.load_ui_values()
        
        # Find the display name that maps to our model type
        model_display_name = next(
            key for key, value in MODEL_TYPES.items() 
            if value == preset["model_type"]
        )
            
        # Get preset description for display
        description = preset.get("description", "")
        
        # Get max values from buckets
        buckets = preset["training_buckets"]
        max_frames = max(frames for frames, _, _ in buckets)
        max_height = max(height for _, height, _ in buckets)
        max_width = max(width for _, _, width in buckets)
        bucket_info = f"\nMaximum video size: {max_frames} frames at {max_width}x{max_height} resolution"
        
        info_text = f"{description}{bucket_info}"
        
        # Return values in the same order as the output components
        # Use preset defaults but preserve user-modified values if they exist
        lora_rank_val = current_state.get("lora_rank") if current_state.get("lora_rank") != preset.get("lora_rank", "128") else preset["lora_rank"]
        lora_alpha_val = current_state.get("lora_alpha") if current_state.get("lora_alpha") != preset.get("lora_alpha", "128") else preset["lora_alpha"]
        num_epochs_val = current_state.get("num_epochs") if current_state.get("num_epochs") != preset.get("num_epochs", 70) else preset["num_epochs"]
        batch_size_val = current_state.get("batch_size") if current_state.get("batch_size") != preset.get("batch_size", 1) else preset["batch_size"]
        learning_rate_val = current_state.get("learning_rate") if current_state.get("learning_rate") != preset.get("learning_rate", 3e-5) else preset["learning_rate"]
        save_iterations_val = current_state.get("save_iterations") if current_state.get("save_iterations") != preset.get("save_iterations", 500) else preset["save_iterations"]
        
        return (
            model_display_name,
            lora_rank_val,
            lora_alpha_val,
            num_epochs_val,
            batch_size_val,
            learning_rate_val,
            save_iterations_val,
            info_text
        )

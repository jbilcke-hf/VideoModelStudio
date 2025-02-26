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

from vms.training_service import TrainingService
from vms.captioning_service import CaptioningService
from vms.splitting_service import SplittingService
from vms.import_service import ImportService
from vms.config import (
    STORAGE_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH,
    TRAINING_PATH, LOG_FILE_PATH, TRAINING_PRESETS, TRAINING_VIDEOS_PATH, MODEL_PATH, OUTPUT_PATH, DEFAULT_CAPTIONING_BOT_INSTRUCTIONS,
    DEFAULT_PROMPT_PREFIX, HF_API_TOKEN, ASK_USER_TO_DUPLICATE_SPACE, MODEL_TYPES, SMALL_TRAINING_BUCKETS
)
from vms.utils import make_archive, count_media_files, format_media_title, is_image_file, is_video_file, validate_model_repo, format_time
from vms.finetrainers_utils import copy_files_to_training_dir, prepare_finetrainers_dataset
from vms.training_log_parser import TrainingLogParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARN)


class VideoTrainerUI:
    def __init__(self):
        self.trainer = TrainingService()
        self.splitter = SplittingService()
        self.importer = ImportService()
        self.captioner = CaptioningService()
        self._should_stop_captioning = False
        self.log_parser = TrainingLogParser()
    
    def update_captioning_buttons_start(self):
        return {
            "run_autocaption_btn": gr.Button(
                interactive=False,
                variant="secondary",
            ),
            "stop_autocaption_btn": gr.Button(
                interactive=True,
                variant="stop",
            ),
            "copy_files_to_training_dir_btn": gr.Button(
                interactive=False,
                variant="secondary",
            )
        }
    
    def update_captioning_buttons_end(self):
        return {
            "run_autocaption_btn": gr.Button(
                interactive=True,
                variant="primary",
            ),
            "stop_autocaption_btn": gr.Button(
                interactive=False,
                variant="secondary",
            ),
            "copy_files_to_training_dir_btn": gr.Button(
                interactive=True,
                variant="primary",
            )
        }

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
        
        print("update_training_ui: training_state = ", training_state)

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

    def handle_training_dataset_select(self, evt: gr.SelectData) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
                    "No file selected"
                ]
                
            # Check both possible locations for the file
            possible_paths = [
                STAGING_PATH / file_name,

                # note: we use to look into this dir for already-captioned clips,
                # but we don't do this anymore
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
                f"Error handling selection: {str(e)}"
            ]
  
    def save_caption_changes(self, preview_caption: str, preview_image: str, preview_video: str, prompt_prefix: str):
        """Save changes to caption"""
        try:
            # Add prefix if not already present 
            if prompt_prefix and not preview_caption.startswith(prompt_prefix):
                full_caption = f"{prompt_prefix}{preview_caption}"
            else:
                full_caption = preview_caption
                
            path = Path(preview_video if preview_video else preview_image)
            file_path = path.with_suffix('') if path.suffix == '.txt' else path
            self.captioner.update_file_caption(file_path, full_caption)
            return gr.update(value="Caption saved successfully!")
        except Exception as e:
            return gr.update(value=f"Error saving caption: {str(e)}")

    def get_model_info(self, model_type: str) -> str:
        """Get information about the selected model type"""
        if model_type == "hunyuan_video":
            return """### HunyuanVideo (LoRA)
    - Best for learning complex video generation patterns
    - Required VRAM: ~47GB minimum
    - Recommended batch size: 1-2
    - Typical training time: 2-4 hours
    - Default resolution: 49x512x768
    - Default LoRA rank: 128"""
                
        elif model_type == "ltx_video":
            return """### LTX-Video (LoRA)
    - Lightweight video model
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
        """Update UI components based on selected preset"""
        preset = TRAINING_PRESETS[preset_name]
        
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
        return (
            model_display_name,
            preset["lora_rank"],
            preset["lora_alpha"],
            preset["num_epochs"],
            preset["batch_size"],
            preset["learning_rate"],
            preset["save_iterations"],
            info_text
        )

    def create_ui(self):
        """Create Gradio interface"""

        with gr.Blocks(title="🎥 Video Model Studio") as app:
            gr.Markdown("# 🎥 Video Model Studio")

            with gr.Tabs() as tabs:
                with gr.TabItem("1️⃣  Import", id="import_tab"):

                    with gr.Row():
                        gr.Markdown("## Automatic splitting and captioning")
                    
                    with gr.Row():
                        enable_automatic_video_split = gr.Checkbox(
                            label="Automatically split videos into smaller clips",
                            info="Note: a clip is a single camera shot, usually a few seconds",
                            value=True,
                            visible=True
                        )
                        enable_automatic_content_captioning = gr.Checkbox(
                            label="Automatically caption photos and videos",
                            info="Note: this uses LlaVA and takes some extra time to load and process",
                            value=False,
                            visible=True,
                        )
                        
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("## Import video files")
                                    gr.Markdown("You can upload either:")
                                    gr.Markdown("- A single MP4 video file")
                                    gr.Markdown("- A ZIP archive containing multiple videos and optional caption files")
                                    gr.Markdown("For ZIP files: Create a folder containing videos (name is not important) and optional caption files with the same name (eg. `some_video.txt` for `some_video.mp4`)")
                                        
                            with gr.Row():
                                files = gr.Files(
                                    label="Upload Images, Videos or ZIP",
                                    #file_count="multiple",
                                    file_types=[".jpg", ".jpeg", ".png", ".webp", ".webp", ".avif", ".heic", ".mp4", ".zip"],
                                    type="filepath"
                                )
               
                        with gr.Column(scale=3):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("## Import a YouTube video")
                                    gr.Markdown("You can also use a YouTube video as reference, by pasting its URL here:")

                            with gr.Row():
                                youtube_url = gr.Textbox(
                                    label="Import YouTube Video",
                                    placeholder="https://www.youtube.com/watch?v=..."
                                )
                            with gr.Row():
                                youtube_download_btn = gr.Button("Download YouTube Video", variant="secondary")
                    with gr.Row():
                        import_status = gr.Textbox(label="Status", interactive=False)


                with gr.TabItem("2️⃣  Split", id="split_tab"):
                    with gr.Row():
                        split_title = gr.Markdown("## Splitting of 0 videos (0 bytes)")
                    
                    with gr.Row():
                        with gr.Column():
                            detect_btn = gr.Button("Split videos into single-camera shots", variant="primary")
                            detect_status = gr.Textbox(label="Status", interactive=False)

                        with gr.Column():

                            video_list = gr.Dataframe(
                                headers=["name", "status"],
                                label="Videos to split",
                                interactive=False,
                                wrap=True,
                                #selection_mode="cell"  # Enable cell selection
                            )
                            
         
                with gr.TabItem("3️⃣  Caption"):
                    with gr.Row():
                        caption_title = gr.Markdown("## Captioning of 0 files (0 bytes)")
                        
                    with gr.Row():
                    
                        with gr.Column():
                            with gr.Row():
                                custom_prompt_prefix = gr.Textbox(
                                    scale=3,
                                    label='Prefix to add to ALL captions (eg. "In the style of TOK, ")',
                                    placeholder="In the style of TOK, ",
                                    lines=2,
                                    value=DEFAULT_PROMPT_PREFIX
                                )
                                captioning_bot_instructions = gr.Textbox(
                                    scale=6,
                                    label="System instructions for the automatic captioning model",
                                    placeholder="Please generate a full description of...",
                                    lines=5,
                                    value=DEFAULT_CAPTIONING_BOT_INSTRUCTIONS
                                )
                            with gr.Row():
                                run_autocaption_btn = gr.Button(
                                    "Automatically fill missing captions",
                                    variant="primary"  # Makes it green by default
                                )
                                copy_files_to_training_dir_btn = gr.Button(
                                    "Copy assets to training directory",
                                    variant="primary"  # Makes it green by default
                                )
                                stop_autocaption_btn = gr.Button(
                                    "Stop Captioning",
                                    variant="stop",  # Red when enabled
                                    interactive=False  # Disabled by default
                                )

                    with gr.Row():
                        with gr.Column():
                            training_dataset = gr.Dataframe(
                                headers=["name", "status"],
                                interactive=False,
                                wrap=True,
                                value=self.list_training_files_to_caption(),
                                row_count=10,  # Optional: set a reasonable row count
                                #selection_mode="cell" 
                            )

                        with gr.Column():
                            preview_video = gr.Video(
                                label="Video Preview",
                                interactive=False,
                                visible=False
                            )
                            preview_image = gr.Image(
                                label="Image Preview",
                                interactive=False,
                                visible=False
                            )
                            preview_caption = gr.Textbox(
                                label="Caption",
                                lines=6,
                                interactive=True
                            )
                            save_caption_btn = gr.Button("Save Caption")
                            preview_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                visible=True
                            )

                with gr.TabItem("4️⃣  Train"):
                    with gr.Row():
                        with gr.Column():

                            with gr.Row():
                                train_title = gr.Markdown("## 0 files available for training (0 bytes)")

                            with gr.Row():
                                with gr.Column():
                                    training_preset = gr.Dropdown(
                                        choices=list(TRAINING_PRESETS.keys()),
                                        label="Training Preset",
                                        value=list(TRAINING_PRESETS.keys())[0]
                                    )
                                preset_info = gr.Markdown()

                            with gr.Row():
                                with gr.Column():
                                    model_type = gr.Dropdown(
                                        choices=list(MODEL_TYPES.keys()),
                                        label="Model Type",
                                        value=list(MODEL_TYPES.keys())[0]
                                    )
                                model_info = gr.Markdown(
                                    value=self.get_model_info(list(MODEL_TYPES.keys())[0])
                                )

                            with gr.Row():
                                lora_rank = gr.Dropdown(
                                    label="LoRA Rank",
                                    choices=["16", "32", "64", "128", "256"],
                                    value="128",
                                    type="value"
                                )
                                lora_alpha = gr.Dropdown(
                                    label="LoRA Alpha",
                                    choices=["16", "32", "64", "128", "256"],
                                    value="128",
                                    type="value"
                                )
                            with gr.Row():
                                num_epochs = gr.Number(
                                    label="Number of Epochs",
                                    value=70,
                                    minimum=1,
                                    precision=0
                                )
                                batch_size = gr.Number(
                                    label="Batch Size",
                                    value=1,
                                    minimum=1,
                                    precision=0
                                )
                            with gr.Row():
                                learning_rate = gr.Number(
                                    label="Learning Rate",
                                    value=2e-5,
                                    minimum=1e-7
                                )
                                save_iterations = gr.Number(
                                    label="Save checkpoint every N iterations",
                                    value=500,
                                    minimum=50,
                                    precision=0,
                                    info="Model will be saved periodically after these many steps"
                                )
                        
                        with gr.Column():
                            with gr.Row():
                                start_btn = gr.Button(
                                    "Start Training",
                                    variant="primary",
                                    interactive=not ASK_USER_TO_DUPLICATE_SPACE
                                )
                                pause_resume_btn = gr.Button(
                                    "Resume Training",
                                    variant="secondary",
                                    interactive=False
                                )
                                stop_btn = gr.Button(
                                    "Stop Training",
                                    variant="stop",
                                    interactive=False
                                )

                            with gr.Row():
                                with gr.Column():
                                    status_box = gr.Textbox(
                                        label="Training Status",
                                        interactive=False,
                                        lines=4
                                    )
                                    with gr.Accordion("See training logs"):
                                        log_box = gr.TextArea(
                                            label="Finetrainers output (see HF Space logs for more details)",
                                            interactive=False,
                                            lines=40,
                                            max_lines=200,
                                            autoscroll=True
                                        )

                with gr.TabItem("5️⃣  Manage"):

                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("## Publishing")
                                gr.Markdown("You model can be pushed to Hugging Face (this will use HF_API_TOKEN)")

                                with gr.Row():

                                    with gr.Column():
                                        repo_id = gr.Textbox(
                                            label="HuggingFace Model Repository",
                                            placeholder="username/model-name",
                                            info="The repository will be created if it doesn't exist"
                                        )
                                        gr.Checkbox(label="Check this to make your model public (ie. visible and downloadable by anyone)", info="You model is private by default"),
                                        global_stop_btn = gr.Button(
                                            "Push my model",
                                            #variant="stop"
                                        )

                        
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("## Storage management")
                                        with gr.Row():
                                            download_dataset_btn = gr.DownloadButton(
                                                "Download dataset",
                                                variant="secondary",
                                                size="lg"
                                            )
                                            download_model_btn = gr.DownloadButton(
                                                "Download model",
                                                variant="secondary",
                                                size="lg"
                                            )


                                with gr.Row():
                                    global_stop_btn = gr.Button(
                                        "Stop everything and delete my data",
                                        variant="stop"
                                    )
                                    global_status = gr.Textbox(
                                        label="Global Status",
                                        interactive=False,
                                        visible=False
                                    )
    

            
            # Event handlers
            def update_model_info(model):
                params = self.get_default_params(MODEL_TYPES[model])
                info = self.get_model_info(MODEL_TYPES[model])
                return {
                    model_info: info,
                    num_epochs: params["num_epochs"],
                    batch_size: params["batch_size"],
                    learning_rate: params["learning_rate"],
                    save_iterations: params["save_iterations"]
                }
            
            def validate_repo(repo_id: str) -> dict:
                validation = validate_model_repo(repo_id)
                if validation["error"]:
                    return gr.update(value=repo_id, error=validation["error"])
                return gr.update(value=repo_id, error=None)
            
            # Connect events 
            model_type.change(
                fn=update_model_info,
                inputs=[model_type],
                outputs=[model_info, num_epochs, batch_size, learning_rate, save_iterations]
            )

            async def on_import_success(enable_splitting, enable_automatic_content_captioning, prompt_prefix):
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

                # Start auto-captioning if enabled
                if enable_automatic_content_captioning:
                    await self.start_caption_generation(
                        DEFAULT_CAPTIONING_BOT_INSTRUCTIONS,
                        prompt_prefix
                    )
                
                return {
                    tabs: gr.Tabs(selected="split_tab"),
                    video_list: videos,
                    detect_status: msg
                }


            async def update_titles_after_import(enable_splitting, enable_automatic_content_captioning, prompt_prefix):
                """Handle post-import updates including titles"""
                import_result = await on_import_success(enable_splitting, enable_automatic_content_captioning, prompt_prefix)
                titles = self.update_titles()
                return (*import_result, *titles)

            files.upload(
                fn=lambda x: self.importer.process_uploaded_files(x),
                inputs=[files],
                outputs=[import_status]
            ).success(
                fn=update_titles_after_import,
                inputs=[enable_automatic_video_split, enable_automatic_content_captioning, custom_prompt_prefix],
                outputs=[
                    tabs, video_list, detect_status,
                    split_title, caption_title, train_title
                ]
            )
            
            youtube_download_btn.click(
                fn=self.importer.download_youtube_video,
                inputs=[youtube_url],
                outputs=[import_status]
            ).success(
                fn=on_import_success,
                inputs=[enable_automatic_video_split, enable_automatic_content_captioning, custom_prompt_prefix],
                outputs=[tabs, video_list, detect_status]
            )

            # Scene detection events
            detect_btn.click(
                fn=self.start_scene_detection,
                inputs=[enable_automatic_video_split],
                outputs=[detect_status]
            )


            # Update button states based on captioning status
            def update_button_states(is_running):
                return {
                    run_autocaption_btn: gr.Button(
                        interactive=not is_running,
                        variant="secondary" if is_running else "primary",
                    ),
                    stop_autocaption_btn: gr.Button(
                        interactive=is_running,
                        variant="secondary",
                    ),
                }
            
            run_autocaption_btn.click(
                fn=self.show_refreshing_status,
                outputs=[training_dataset]
            ).then(
                fn=lambda: self.update_captioning_buttons_start(),
                outputs=[run_autocaption_btn, stop_autocaption_btn, copy_files_to_training_dir_btn]
            ).then(
                fn=self.start_caption_generation,
                inputs=[captioning_bot_instructions, custom_prompt_prefix],
                outputs=[training_dataset],
            ).then(
                fn=lambda: self.update_captioning_buttons_end(),
                outputs=[run_autocaption_btn, stop_autocaption_btn, copy_files_to_training_dir_btn]
            )
            
            copy_files_to_training_dir_btn.click(
                fn=self.copy_files_to_training_dir,
                inputs=[custom_prompt_prefix]
            )
            stop_autocaption_btn.click(
                fn=self.stop_captioning,
                outputs=[training_dataset, run_autocaption_btn, stop_autocaption_btn, copy_files_to_training_dir_btn]
            )
            training_dataset.select(
                fn=self.handle_training_dataset_select,
                outputs=[preview_image, preview_video, preview_caption, preview_status]
            )

            save_caption_btn.click(
                fn=self.save_caption_changes,
                inputs=[preview_caption, preview_image, preview_video, custom_prompt_prefix],
                outputs=[preview_status]
            ).success(
                fn=self.list_training_files_to_caption,
                outputs=[training_dataset]
            )
            
            training_preset.change(
                fn=self.update_training_params,
                inputs=[training_preset],
                outputs=[
                    model_type, lora_rank, lora_alpha, 
                    num_epochs, batch_size, learning_rate, 
                    save_iterations, preset_info
                ]
            )

            # Training control events
            start_btn.click(
                fn=lambda preset, model_type, *args: (
                    self.log_parser.reset(),
                    self.trainer.start_training(
                        MODEL_TYPES[model_type],
                        *args,
                        preset_name=preset
                    )
                ),
                inputs=[
                    training_preset,
                    model_type,
                    lora_rank,
                    lora_alpha,
                    num_epochs,
                    batch_size,
                    learning_rate,
                    save_iterations,
                    repo_id
                ],
                outputs=[status_box, log_box]
            ).success(
                fn=self.get_latest_status_message_logs_and_button_labels,
                outputs=[status_box, log_box, start_btn, stop_btn, pause_resume_btn]
            )

            pause_resume_btn.click(
                fn=self.handle_pause_resume,
                outputs=[status_box, log_box, start_btn, stop_btn, pause_resume_btn]
            )

            stop_btn.click(
                fn=self.handle_stop,
                outputs=[status_box, log_box, start_btn, stop_btn, pause_resume_btn]
            )

            def handle_global_stop():
                result = self.stop_all_and_clear()
                # Update all relevant UI components
                status = result["status"]
                details = "\n".join(f"{k}: {v}" for k, v in result["details"].items())
                full_status = f"{status}\n\nDetails:\n{details}"
                
                # Get fresh lists after cleanup
                videos = self.splitter.list_unprocessed_videos()
                clips = self.list_training_files_to_caption()
                
                return {
                    global_status: gr.update(value=full_status, visible=True),
                    video_list: videos,
                    training_dataset: clips,
                    status_box: "Training stopped and data cleared",
                    log_box: "",
                    detect_status: "Scene detection stopped",
                    import_status: "All data cleared",
                    preview_status: "Captioning stopped"
                }
            
            download_dataset_btn.click(
                fn=self.trainer.create_training_dataset_zip,
                outputs=[download_dataset_btn]
            )

            download_model_btn.click(
                fn=self.trainer.get_model_output_safetensors,
                outputs=[download_model_btn]
            )

            global_stop_btn.click(
                fn=handle_global_stop,
                outputs=[
                    global_status,
                    video_list,
                    training_dataset,
                    status_box,
                    log_box,
                    detect_status,
                    import_status,
                    preview_status
                ]
            )

            # Auto-refresh timers
            app.load(
                fn=lambda: (
                    self.refresh_dataset()
                ),
                outputs=[
                    video_list, training_dataset
                ]
            )
            
            timer = gr.Timer(value=1)
            timer.tick(
                fn=lambda: (
                    self.get_latest_status_message_logs_and_button_labels()
                ),
                outputs=[
                    status_box,
                    log_box,
                    start_btn,
                    stop_btn,
                    pause_resume_btn
                ]
            )

            timer = gr.Timer(value=5)
            timer.tick(
                fn=lambda: (
                    self.refresh_dataset()
                ),
                outputs=[
                    video_list, training_dataset
                ]
            )

            timer = gr.Timer(value=6)
            timer.tick(
                fn=lambda: self.update_titles(),
                outputs=[
                    split_title, caption_title, train_title
                ]
            )

        return app

def create_app():
    if ASK_USER_TO_DUPLICATE_SPACE:
        with gr.Blocks() as app:
            gr.Markdown("""# Finetrainers UI

This Hugging Face space needs to be duplicated to your own billing account to work.

Click the 'Duplicate Space' button at the top of the page to create your own copy.

It is recommended to use a Nvidia L40S and a persistent storage space.
To avoid overpaying for your space, you can configure the auto-sleep settings to fit your personal budget.""")
        return app

    ui = VideoTrainerUI()
    return ui.create_ui()

if __name__ == "__main__":
    app = create_app()

    allowed_paths = [
        str(STORAGE_PATH),  # Base storage
        str(VIDEOS_TO_SPLIT_PATH),
        str(STAGING_PATH), 
        str(TRAINING_PATH),
        str(TRAINING_VIDEOS_PATH),
        str(MODEL_PATH),
        str(OUTPUT_PATH)
    ]
    app.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        allowed_paths=allowed_paths
    )
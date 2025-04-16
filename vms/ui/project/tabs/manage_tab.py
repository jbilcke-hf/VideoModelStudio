"""
Manage tab for Video Model Studio UI
"""

import gradio as gr
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from vms.utils import BaseTab, validate_model_repo
from vms.config import (
    HF_API_TOKEN, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, 
    USE_LARGE_DATASET
)

logger = logging.getLogger(__name__)

class ManageTab(BaseTab):
    """Manage tab for storage management and model publication"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "manage_tab"
        self.title = "5️⃣ Storage"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Manage tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 🏦 Backup your model")
                    gr.Markdown("There is currently a bug, you might have to click multiple times to trigger a download.")

                    with gr.Row():
                        self.components["download_dataset_btn"] = gr.DownloadButton(
                            "📦 Download training dataset (.zip)",
                            variant="secondary",
                            size="lg",
                            visible=not USE_LARGE_DATASET
                        )
                        # If we have a large dataset, display a message explaining why download is disabled
                        if USE_LARGE_DATASET:
                            gr.Markdown("📦 Training dataset download disabled for large datasets")
                            
                        self.components["download_model_btn"] = gr.DownloadButton(
                            "🧠 Download weights (.safetensors)",
                            variant="secondary",
                            size="lg"
                        )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📡 Publish your model")
                    gr.Markdown("You model can be pushed to Hugging Face (this will use HF_API_TOKEN)")

            with gr.Row():
                with gr.Column():
                    self.components["repo_id"] = gr.Textbox(
                        label="HuggingFace Model Repository",
                        placeholder="username/model-name",
                        info="The repository will be created if it doesn't exist"
                    )
                    self.components["make_public"] = gr.Checkbox(
                        label="Check this to make your model public (ie. visible and downloadable by anyone)",
                        info="You model is private by default"
                    )
                    self.components["push_model_btn"] = gr.Button(
                        "Push my model"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("## ♻️ Delete your data")
                    gr.Markdown("Make sure you have made a backup first.")
                    gr.Markdown("If you are deleting because of a bug, remember you can use the Developer Mode on HF to inspect the working directory (in /data or .data)")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧽 Delete specific data")
                    gr.Markdown("You can selectively delete either the dataset and/or the last model data.")

            with gr.Row():
                with gr.Column(scale=1):
                    self.components["delete_dataset_btn"] = gr.Button(
                        "🚨 Delete dataset (images, video, captions)",
                        variant="secondary"
                    )
                    self.components["delete_dataset_status"] = gr.Textbox(
                        label="Delete Dataset Status",
                        interactive=False,
                        visible=False
                    )
                
                with gr.Column(scale=1):
                    self.components["delete_model_btn"] = gr.Button(
                        "🚨 Delete model (checkpoints, weights, config)",
                        variant="secondary"
                    )
                    self.components["delete_model_status"] = gr.Textbox(
                        label="Delete Model Status",
                        interactive=False,
                        visible=False
                    )
                
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ☢️ Nuke all project data")
                    gr.Markdown("This will nuke the original dataset (all images, videos and captions), the training dataset, and the model outputs (weights, checkpoints, settings). So use with care!")

            with gr.Row():
                self.components["global_stop_btn"] = gr.Button(
                    "🚨 Delete all project data and models (are you sure?!)",
                    variant="stop"
                )
                self.components["global_status"] = gr.Textbox(
                    label="Global Status",
                    interactive=False,
                    visible=False
                )
        
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Repository ID validation
        self.components["repo_id"].change(
            fn=self.validate_repo,
            inputs=[self.components["repo_id"]],
            outputs=[self.components["repo_id"]]
        )
        
        # Download buttons
        self.components["download_dataset_btn"].click(
            fn=self.app.training.create_training_dataset_zip,
            outputs=[self.components["download_dataset_btn"]]
        )

        self.components["download_model_btn"].click(
            fn=self.app.training.get_model_output_safetensors,
            outputs=[self.components["download_model_btn"]]
        )
        
        # New delete dataset button
        self.components["delete_dataset_btn"].click(
            fn=self.delete_dataset,
            outputs=[
                self.components["delete_dataset_status"],
                self.app.tabs["caption_tab"].components["training_dataset"]
            ]
        )
        
        # New delete model button
        self.components["delete_model_btn"].click(
            fn=self.delete_model,
            outputs=[
                self.components["delete_model_status"],
                self.app.tabs["train_tab"].components["status_box"]
            ]
        )
        
        # Global stop button
        self.components["global_stop_btn"].click(
            fn=self.handle_global_stop,
            outputs=[
                self.components["global_status"],
                self.app.tabs["caption_tab"].components["training_dataset"],
                self.app.tabs["train_tab"].components["status_box"],
                self.app.tabs["train_tab"].components["log_box"],
                self.app.tabs["import_tab"].components["import_status"],
                self.app.tabs["caption_tab"].components["preview_status"]
            ]
        )
        
        # Push model button 
        self.components["push_model_btn"].click(
            fn=lambda repo_id: self.upload_to_hub(repo_id),
            inputs=[self.components["repo_id"]],
            outputs=[self.components["global_status"]]
        )
        
    def validate_repo(self, repo_id: str) -> gr.update:
        """Validate repository ID for HuggingFace Hub"""
        validation = validate_model_repo(repo_id)
        if validation["error"]:
            return gr.update(value=repo_id, error=validation["error"])
        return gr.update(value=repo_id, error=None)
        
    def upload_to_hub(self, repo_id: str) -> str:
        """Upload model to HuggingFace Hub"""
        if not repo_id:
            return "Error: Repository ID is required"
        
        # Validate repository name
        validation = validate_model_repo(repo_id)
        if validation["error"]:
            return f"Error: {validation['error']}"
        
        # Check if we have a model to upload
        if not self.app.training.get_model_output_safetensors():
            return "Error: No model found to upload"
        
        # Upload model to hub
        success = self.app.training.upload_to_hub(self.app.output_path, repo_id)
        
        if success:
            return f"Successfully uploaded model to {repo_id}"
        else:
            return f"Failed to upload model to {repo_id}"
    
    def delete_dataset(self):
        """Delete dataset files (images, videos, captions)"""
        status_messages = {}
        
        try:
            # Stop captioning if running
            if self.app.captioning:
                self.app.captioning.stop_captioning()
                status_messages["captioning"] = "Captioning stopped"
            
            # Stop scene detection if running
            if self.app.splitting.is_processing():
                self.app.splitting.processing = False
                status_messages["splitting"] = "Scene detection stopped"
            
            # Clear dataset directories
            for path in [VIDEOS_TO_SPLIT_PATH, STAGING_PATH, self.app.training_videos_path, self.app.training_path]:
                if path.exists():
                    try:
                        shutil.rmtree(path)
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        status_messages[f"clear_{path.name}"] = f"Error clearing {path.name}: {str(e)}"
                    else:
                        status_messages[f"clear_{path.name}"] = f"Cleared {path.name}"
            
            # Reset any relevant persistent state
            self.app.tabs["caption_tab"]._should_stop_captioning = True
            self.app.splitting.processing = False
            
            # Format response
            details = "\n".join(f"{k}: {v}" for k, v in status_messages.items())
            message = f"Dataset deleted successfully\n\nDetails:\n{details}"
            
            # Get fresh lists after cleanup
            clips = self.app.tabs["caption_tab"].list_training_files_to_caption()
            
            return gr.update(value=message, visible=True), clips
            
        except Exception as e:
            error_message = f"Error deleting dataset: {str(e)}\n\nDetails:\n{status_messages}"
            return gr.update(value=error_message, visible=True), self.app.tabs["caption_tab"].list_training_files_to_caption()
    
    def delete_model(self):
        """Delete model files (checkpoints, weights, configuration)"""
        status_messages = {}
        
        try:
            # Stop training if running
            if self.app.training.is_training_running():
                training_result = self.app.training.stop_training()
                status_messages["training"] = training_result["status"]
            
            # Clear model output directory
            if self.app.output_path.exists():
                try:
                    shutil.rmtree(self.app.output_path)
                    self.app.output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    status_messages[f"clear_{self.app.output_path.name}"] = f"Error clearing {self.app.output_path.name}: {str(e)}"
                else:
                    status_messages[f"clear_{self.app.output_path.name}"] = f"Cleared {self.app.output_path.name}"
            
            # Properly close logging before clearing log file
            if self.app.training.file_handler:
                self.app.training.file_handler.close()
                logger.removeHandler(self.app.training.file_handler)
                self.app.training.file_handler = None
                
            if self.app.log_file_path.exists():
                self.app.log_file_path.unlink()
            
            # Reset training UI state
            self.app.training.setup_logging()
            
            # Format response
            details = "\n".join(f"{k}: {v}" for k, v in status_messages.items())
            message = f"Model deleted successfully\n\nDetails:\n{details}"
            
            return gr.update(value=message, visible=True), "Model files have been deleted"
            
        except Exception as e:
            error_message = f"Error deleting model: {str(e)}\n\nDetails:\n{status_messages}"
            return gr.update(value=error_message, visible=True), f"Error deleting model: {str(e)}"
            
    def handle_global_stop(self):
        """Handle the global stop button click"""
        result = self.stop_all_and_clear()
        
        # Format the details for display
        status = result["status"]
        details = "\n".join(f"{k}: {v}" for k, v in result["details"].items())
        full_status = f"{status}\n\nDetails:\n{details}"
        
        # Get fresh lists after cleanup
        clips = self.app.tabs["caption_tab"].list_training_files_to_caption()
        
        return {
            self.components["global_status"]: gr.update(value=full_status, visible=True),
            self.app.tabs["caption_tab"].components["training_dataset"]: clips,
            self.app.tabs["train_tab"].components["status_box"]: "Training stopped and data cleared",
            self.app.tabs["train_tab"].components["log_box"]: "",
            self.app.tabs["import_tab"].components["import_status"]: "All data cleared",
            self.app.tabs["caption_tab"].components["preview_status"]: "Captioning stopped"
        }
        
    def stop_all_and_clear(self) -> Dict[str, str]:
        """Stop all running processes and clear data
        
        Returns:
            Dict with status messages for different components
        """
        status_messages = {}
        
        try:
            # Stop training if running
            if self.app.training.is_training_running():
                training_result = self.app.training.stop_training()
                status_messages["training"] = training_result["status"]
            
            # Stop captioning if running
            if self.app.captioning:
                self.app.captioning.stop_captioning()
                status_messages["captioning"] = "Captioning stopped"
            
            # Stop scene detection if running
            if self.app.splitting.is_processing():
                self.app.splitting.processing = False
                status_messages["splitting"] = "Scene detection stopped"
            
            # Properly close logging before clearing log file
            if self.app.training.file_handler:
                self.app.training.file_handler.close()
                logger.removeHandler(self.app.training.file_handler)
                self.app.training.file_handler = None
                
            if self.app.log_file_path.exists():
                self.app.log_file_path.unlink()
            
            # Clear all data directories
            for path in [VIDEOS_TO_SPLIT_PATH, STAGING_PATH, self.app.training_videos_path, self.app.training_path, self.app.output_path]:
                if path.exists():
                    try:
                        shutil.rmtree(path)
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        status_messages[f"clear_{path.name}"] = f"Error clearing {path.name}: {str(e)}"
                    else:
                        status_messages[f"clear_{path.name}"] = f"Cleared {path.name}"
            
            # Reset any persistent state
            self.app.tabs["caption_tab"]._should_stop_captioning = True
            self.app.splitting.processing = False
            
            # Recreate logging setup
            self.app.training.setup_logging()
            
            return {
                "status": "All processes stopped and data cleared",
                "details": status_messages
            }
            
        except Exception as e:
            return {
                "status": f"Error during cleanup: {str(e)}",
                "details": status_messages
            }
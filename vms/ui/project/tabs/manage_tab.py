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
    HF_API_TOKEN, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, TRAINING_VIDEOS_PATH, 
    TRAINING_PATH, MODEL_PATH, OUTPUT_PATH, LOG_FILE_PATH
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
                    gr.Markdown("## Download your model")
                    gr.Markdown("There is currently a bug, you might have to click multiple times to trigger a download.")

                    with gr.Row():
                        self.components["download_dataset_btn"] = gr.DownloadButton(
                            "Download training dataset",
                            variant="secondary",
                            size="lg"
                        )
                        self.components["download_model_btn"] = gr.DownloadButton(
                            "Download model weights",
                            variant="secondary",
                            size="lg"
                        )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Publish your model")
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
                    gr.Markdown("## Delete your model")
                    gr.Markdown("If something went wrong, you can trigger a full reset (model shutdown + data destruction).")
                    gr.Markdown("Make sure you have made a backup first.")
                    gr.Markdown("If you are deleting because of a bug, remember you can use the Developer Mode on HF to inspect the working directory (in /data or .data)")

            with gr.Row():
                self.components["global_stop_btn"] = gr.Button(
                    "Stop everything and delete my data",
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
        success = self.app.training.upload_to_hub(OUTPUT_PATH, repo_id)
        
        if success:
            return f"Successfully uploaded model to {repo_id}"
        else:
            return f"Failed to upload model to {repo_id}"
            
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
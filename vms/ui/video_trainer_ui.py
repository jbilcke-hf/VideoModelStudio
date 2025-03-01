import platform
import gradio as gr
from pathlib import Path
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union, Tuple

from ..services import TrainingService, CaptioningService, SplittingService, ImportService
from ..config import (
    STORAGE_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH,
    TRAINING_PATH, LOG_FILE_PATH, TRAINING_PRESETS, TRAINING_VIDEOS_PATH, MODEL_PATH, OUTPUT_PATH,
    MODEL_TYPES, SMALL_TRAINING_BUCKETS
)
from ..utils import count_media_files, format_media_title, TrainingLogParser
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
        
        # Recovery status from any interrupted training
        recovery_result = self.trainer.recover_interrupted_training()
        self.recovery_status = recovery_result.get("status", "unknown")
        self.ui_updates = recovery_result.get("ui_updates", {})
        
        # Initialize log parser
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
            fn=self.tabs["train_tab"].get_latest_status_message_logs_and_button_labels,
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
    
    def initialize_app_state(self):
        """Initialize all app state in one function to ensure correct output count"""
        # Get dataset info
        video_list = self.tabs["split_tab"].list_unprocessed_videos()
        training_dataset = self.tabs["caption_tab"].list_training_files_to_caption()
        
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
    
    def refresh_dataset(self):
        """Refresh all dynamic lists and training state"""
        video_list = self.tabs["split_tab"].list_unprocessed_videos()
        training_dataset = self.tabs["caption_tab"].list_training_files_to_caption()

        return (
            video_list,
            training_dataset
        )
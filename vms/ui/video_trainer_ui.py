import platform
import gradio as gr
from pathlib import Path
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union, Tuple

from ..services import TrainingService, CaptioningService, SplittingService, ImportingService, PreviewingService, MonitoringService
from ..config import (
    STORAGE_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, OUTPUT_PATH,
    TRAINING_PATH, LOG_FILE_PATH, TRAINING_PRESETS, TRAINING_VIDEOS_PATH, MODEL_PATH, OUTPUT_PATH,
    MODEL_TYPES, SMALL_TRAINING_BUCKETS, TRAINING_TYPES,
    DEFAULT_NB_TRAINING_STEPS, DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
    DEFAULT_BATCH_SIZE, DEFAULT_CAPTION_DROPOUT_P,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK_STR, DEFAULT_LORA_ALPHA_STR,
    DEFAULT_SEED,
    DEFAULT_NUM_GPUS,
    DEFAULT_MAX_GPUS,
    DEFAULT_PRECOMPUTATION_ITEMS,
    DEFAULT_NB_TRAINING_STEPS,
    DEFAULT_NB_LR_WARMUP_STEPS
)
from ..utils import (
    get_recommended_precomputation_items,
    count_media_files,
    format_media_title,
    TrainingLogParser
)
from ..tabs import ImportTab, SplitTab, CaptionTab, TrainTab, MonitorTab, PreviewTab, ManageTab

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARN)

class VideoTrainerUI:       
    def __init__(self):
        """Initialize services and tabs"""
        # Initialize core services
        self.training = TrainingService(self)
        self.splitting = SplittingService()
        self.importing = ImportingService()
        self.captioning = CaptioningService()
        self.monitoring = MonitoringService()
        self.previewing = PreviewingService()

        # Start the monitoring service on app creation
        self.monitoring.start_monitoring()
    
        # Recovery status from any interrupted training
        recovery_result = self.training.recover_interrupted_training()
        # Add null check for recovery_result
        if recovery_result is None:
            recovery_result = {"status": "unknown", "ui_updates": {}}
        
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

        # Log recovery status
        logger.info(f"Initialization complete. Recovery status: {self.recovery_status}")
    
    def add_periodic_callback(self, callback_fn, interval=1.0):
        """Add a periodic callback function to the UI
        
        Args:
            callback_fn: Function to call periodically
            interval: Time in seconds between calls (default: 1.0)
        """
        try:
            # Store a reference to the callback function
            if not hasattr(self, "_periodic_callbacks"):
                self._periodic_callbacks = []
            
            self._periodic_callbacks.append(callback_fn)
            
            # Add the callback to the Gradio app
            self.app.add_callback(
                interval,  # Interval in seconds
                callback_fn,  # Function to call
                inputs=None,  # No inputs needed
                outputs=list(self.components.values())  # All components as possible outputs
            )
            
            logger.info(f"Added periodic callback {callback_fn.__name__} with interval {interval}s")
        except Exception as e:
            logger.error(f"Error adding periodic callback: {e}", exc_info=True)
            
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
                self.tabs["monitor_tab"] = MonitorTab(self)
                self.tabs["preview_tab"] = PreviewTab(self)
                self.tabs["manage_tab"] = ManageTab(self)
                
                # Create tab UI components
                for tab_id, tab_obj in self.tabs.items():
                    tab_obj.create(self.tabs_component)
            
            # Connect event handlers
            for tab_id, tab_obj in self.tabs.items():
                tab_obj.connect_events()
            
            # app-level timers for auto-refresh functionality
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
                    self.tabs["train_tab"].components["training_type"],
                    self.tabs["train_tab"].components["lora_rank"],
                    self.tabs["train_tab"].components["lora_alpha"],
                    self.tabs["train_tab"].components["train_steps"],
                    self.tabs["train_tab"].components["batch_size"],
                    self.tabs["train_tab"].components["learning_rate"],
                    self.tabs["train_tab"].components["save_iterations"],
                    self.tabs["train_tab"].components["current_task_box"],
                    self.tabs["train_tab"].components["num_gpus"],
                    self.tabs["train_tab"].components["precomputation_items"],
                    self.tabs["train_tab"].components["lr_warmup_steps"]
                ]
            )
            
        return app
    
    def _add_timers(self):
        """Add auto-refresh timers to the UI"""
        # Status update timer for text components (every 1 second)
        status_timer = gr.Timer(value=1)
        status_timer.tick(
            fn=self.tabs["train_tab"].get_status_updates,  # Use a new function that returns appropriate updates
            outputs=[
                self.tabs["train_tab"].components["status_box"],
                self.tabs["train_tab"].components["log_box"],
                self.tabs["train_tab"].components["current_task_box"] if "current_task_box" in self.tabs["train_tab"].components else None
            ]
        )
        
        # Button update timer for button components (every 1 second)
        button_timer = gr.Timer(value=1)
        button_outputs = [
            self.tabs["train_tab"].components["start_btn"],
            self.tabs["train_tab"].components["stop_btn"]
        ]
        
        # Add delete_checkpoints_btn or pause_resume_btn as the third button
        if "delete_checkpoints_btn" in self.tabs["train_tab"].components:
            button_outputs.append(self.tabs["train_tab"].components["delete_checkpoints_btn"])
        elif "pause_resume_btn" in self.tabs["train_tab"].components:
            button_outputs.append(self.tabs["train_tab"].components["pause_resume_btn"])
        
        button_timer.tick(
            fn=self.tabs["train_tab"].get_button_updates,  # Use a new function for button-specific updates
            outputs=button_outputs
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
        
        # Get button states based on recovery status
        button_states = self.get_initial_button_states()
        start_btn = button_states[0]
        stop_btn = button_states[1]
        delete_checkpoints_btn = button_states[2]  # This replaces pause_resume_btn in the response tuple
        
        # Get UI form values - possibly from the recovery
        if self.recovery_status in ["recovered", "ready_to_recover", "running"] and "ui_updates" in self.state["recovery_result"]:
            recovery_ui = self.state["recovery_result"]["ui_updates"]
            
            # If we recovered training parameters from the original session
            ui_state = {}
            
            # Handle model_type specifically - could be internal or display name
            if "model_type" in recovery_ui:
                model_type_value = recovery_ui["model_type"]
                
                # Remove " (LoRA)" suffix if present
                if " (LoRA)" in model_type_value:
                    model_type_value = model_type_value.replace(" (LoRA)", "")
                    logger.info(f"Removed (LoRA) suffix from model type: {model_type_value}")
                
                # If it's an internal name, convert to display name
                if model_type_value not in MODEL_TYPES:
                    # Find the display name for this internal model type
                    for display_name, internal_name in MODEL_TYPES.items():
                        if internal_name == model_type_value:
                            model_type_value = display_name
                            logger.info(f"Converted internal model type '{recovery_ui['model_type']}' to display name '{model_type_value}'")
                            break
                
                ui_state["model_type"] = model_type_value
            
            # Handle training_type
            if "training_type" in recovery_ui:
                training_type_value = recovery_ui["training_type"]
                
                # If it's an internal name, convert to display name
                if training_type_value not in TRAINING_TYPES:
                    for display_name, internal_name in TRAINING_TYPES.items():
                        if internal_name == training_type_value:
                            training_type_value = display_name
                            logger.info(f"Converted internal training type '{recovery_ui['training_type']}' to display name '{training_type_value}'")
                            break
                
                ui_state["training_type"] = training_type_value
            
            # Copy other parameters
            for param in ["lora_rank", "lora_alpha", "train_steps", 
                        "batch_size", "learning_rate", "save_iterations", "training_preset"]:
                if param in recovery_ui:
                    ui_state[param] = recovery_ui[param]
            
            # Merge with existing UI state if needed
            if ui_state:
                current_state = self.load_ui_values()
                current_state.update(ui_state)
                self.training.save_ui_state(current_state)
                logger.info(f"Updated UI state from recovery: {ui_state}")
        
        # Load values (potentially with recovery updates applied)
        ui_state = self.load_ui_values()
        
        # Ensure model_type is a valid display name
        model_type_val = ui_state.get("model_type", list(MODEL_TYPES.keys())[0])
        # Remove " (LoRA)" suffix if present
        if " (LoRA)" in model_type_val:
            model_type_val = model_type_val.replace(" (LoRA)", "")
            logger.info(f"Removed (LoRA) suffix from model type: {model_type_val}")
        
        # Ensure it's a valid model type in the dropdown
        if model_type_val not in MODEL_TYPES:
            # Convert from internal to display name or use default
            model_type_found = False
            for display_name, internal_name in MODEL_TYPES.items():
                if internal_name == model_type_val:
                    model_type_val = display_name
                    model_type_found = True
                    break
            # If still not found, use the first model type
            if not model_type_found:
                model_type_val = list(MODEL_TYPES.keys())[0]
                logger.warning(f"Invalid model type '{model_type_val}', using default: {model_type_val}")
        
        # Ensure training_type is a valid display name
        training_type_val = ui_state.get("training_type", list(TRAINING_TYPES.keys())[0])
        if training_type_val not in TRAINING_TYPES:
            # Convert from internal to display name or use default
            training_type_found = False
            for display_name, internal_name in TRAINING_TYPES.items():
                if internal_name == training_type_val:
                    training_type_val = display_name
                    training_type_found = True
                    break
            # If still not found, use the first training type
            if not training_type_found:
                training_type_val = list(TRAINING_TYPES.keys())[0]
                logger.warning(f"Invalid training type '{training_type_val}', using default: {training_type_val}")
        
        # Validate training preset
        training_preset = ui_state.get("training_preset", list(TRAINING_PRESETS.keys())[0])
        if training_preset not in TRAINING_PRESETS:
            training_preset = list(TRAINING_PRESETS.keys())[0]
            logger.warning(f"Invalid training preset '{training_preset}', using default: {training_preset}")
        
        # Rest of the function remains unchanged
        lora_rank_val = ui_state.get("lora_rank", DEFAULT_LORA_RANK_STR)
        lora_alpha_val = ui_state.get("lora_alpha", DEFAULT_LORA_ALPHA_STR)
        batch_size_val = int(ui_state.get("batch_size", DEFAULT_BATCH_SIZE))
        learning_rate_val = float(ui_state.get("learning_rate", DEFAULT_LEARNING_RATE))
        save_iterations_val = int(ui_state.get("save_iterations", DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS))
        
        # Update for new UI components
        num_gpus_val = int(ui_state.get("num_gpus", DEFAULT_NUM_GPUS))
        
        # Calculate recommended precomputation items based on video count
        video_count = len(list(TRAINING_VIDEOS_PATH.glob('*.mp4')))
        recommended_precomputation = get_recommended_precomputation_items(video_count, num_gpus_val)
        precomputation_items_val = int(ui_state.get("precomputation_items", recommended_precomputation))
        
        # Ensure warmup steps are not more than training steps
        train_steps_val = int(ui_state.get("train_steps", DEFAULT_NB_TRAINING_STEPS))
        default_warmup = min(DEFAULT_NB_LR_WARMUP_STEPS, int(train_steps_val * 0.2))
        lr_warmup_steps_val = int(ui_state.get("lr_warmup_steps", default_warmup))
        
        # Ensure warmup steps <= training steps
        lr_warmup_steps_val = min(lr_warmup_steps_val, train_steps_val)
        
        # Initial current task value
        current_task_val = ""
        if hasattr(self, 'log_parser') and self.log_parser:
            current_task_val = self.log_parser.get_current_task_display()
        
        # Return all values in the exact order expected by outputs
        return (
            video_list, 
            training_dataset,
            start_btn, 
            stop_btn, 
            delete_checkpoints_btn,
            training_preset, 
            model_type_val,
            training_type_val,
            lora_rank_val, 
            lora_alpha_val,
            train_steps_val, 
            batch_size_val, 
            learning_rate_val, 
            save_iterations_val,
            current_task_val,
            num_gpus_val,
            precomputation_items_val,
            lr_warmup_steps_val
        )

    def initialize_ui_from_state(self):
        """Initialize UI components from saved state"""
        ui_state = self.load_ui_values()
        
        # Return values in order matching the outputs in app.load
        return (
            ui_state.get("training_preset", list(TRAINING_PRESETS.keys())[0]),
            ui_state.get("model_type", list(MODEL_TYPES.keys())[0]),
            ui_state.get("training_type", list(TRAINING_TYPES.keys())[0]),
            ui_state.get("lora_rank", DEFAULT_LORA_RANK_STR),
            ui_state.get("lora_alpha", DEFAULT_LORA_ALPHA_STR),
            ui_state.get("train_steps", DEFAULT_NB_TRAINING_STEPS),
            ui_state.get("batch_size", DEFAULT_BATCH_SIZE),
            ui_state.get("learning_rate", DEFAULT_LEARNING_RATE),
            ui_state.get("save_iterations", DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS)
        )

    def update_ui_state(self, **kwargs):
        """Update UI state with new values"""
        current_state = self.training.load_ui_state()
        current_state.update(kwargs)
        self.training.save_ui_state(current_state)
        # Don't return anything to avoid Gradio warnings
        return None

    def load_ui_values(self):
        """Load UI state values for initializing form fields"""
        ui_state = self.training.load_ui_state()
        
        # Ensure proper type conversion for numeric values
        ui_state["lora_rank"] = ui_state.get("lora_rank", DEFAULT_LORA_RANK_STR)
        ui_state["lora_alpha"] = ui_state.get("lora_alpha", DEFAULT_LORA_ALPHA_STR)
        ui_state["train_steps"] = int(ui_state.get("train_steps", DEFAULT_NB_TRAINING_STEPS))
        ui_state["batch_size"] = int(ui_state.get("batch_size", DEFAULT_BATCH_SIZE))
        ui_state["learning_rate"] = float(ui_state.get("learning_rate", DEFAULT_LEARNING_RATE))
        ui_state["save_iterations"] = int(ui_state.get("save_iterations", DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS))
        
        return ui_state
    
    # Add this new method to get initial button states:
    def get_initial_button_states(self):
        """Get the initial states for training buttons based on recovery status"""
        recovery_result = self.state.get("recovery_result") or self.training.recover_interrupted_training()
        ui_updates = recovery_result.get("ui_updates", {})
        
        # Check for checkpoints to determine start button text
        has_checkpoints = len(list(OUTPUT_PATH.glob("checkpoint-*"))) > 0
        
        # Default button states if recovery didn't provide any
        if not ui_updates or not ui_updates.get("start_btn"):
            is_training = self.training.is_training_running()
            
            if is_training:
                # Active training detected
                start_btn_props = {"interactive": False, "variant": "secondary", "value": "Continue Training" if has_checkpoints else "Start Training"}
                stop_btn_props = {"interactive": True, "variant": "primary", "value": "Stop at Last Checkpoint"}
                delete_btn_props = {"interactive": False, "variant": "stop", "value": "Delete All Checkpoints"}
            else:
                # No active training
                start_btn_props = {"interactive": True, "variant": "primary", "value": "Continue Training" if has_checkpoints else "Start Training"}
                stop_btn_props = {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"}
                delete_btn_props = {"interactive": has_checkpoints, "variant": "stop", "value": "Delete All Checkpoints"}
        else:
            # Use button states from recovery
            start_btn_props = ui_updates.get("start_btn", {"interactive": True, "variant": "primary", "value": "Start Training"})
            stop_btn_props = ui_updates.get("stop_btn", {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"})
            delete_btn_props = ui_updates.get("delete_checkpoints_btn", {"interactive": has_checkpoints, "variant": "stop", "value": "Delete All Checkpoints"})
        
        # Return button states in the correct order
        return (
            gr.Button(**start_btn_props),
            gr.Button(**stop_btn_props),
            gr.Button(**delete_btn_props)
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
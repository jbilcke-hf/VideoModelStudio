import platform
import gradio as gr
from pathlib import Path
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union, Tuple

from vms.config import (
    STORAGE_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, OUTPUT_PATH,
    TRAINING_PATH, LOG_FILE_PATH, TRAINING_PRESETS, TRAINING_VIDEOS_PATH, MODEL_PATH, OUTPUT_PATH,
    MODEL_TYPES, SMALL_TRAINING_BUCKETS, TRAINING_TYPES, MODEL_VERSIONS,
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
from vms.utils import (
    get_recommended_precomputation_items,
    count_media_files,
    format_media_title,
    TrainingLogParser
)

from vms.ui.project.services import (
    TrainingService, CaptioningService, SplittingService, ImportingService, PreviewingService
)
from vms.ui.project.tabs import (
    ImportTab, CaptionTab, TrainTab, PreviewTab, ManageTab
)

from vms.ui.monitoring.services import (
    MonitoringService
)

from vms.ui.monitoring.tabs import (
    GeneralTab
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARN)

class AppUI:       
    def __init__(self):
        """Initialize services and tabs"""
        # Project view
        self.training = TrainingService(self)
        self.splitting = SplittingService()
        self.importing = ImportingService()
        self.captioning = CaptioningService()
        self.previewing = PreviewingService()

        # Monitoring view
        self.monitoring = MonitoringService()
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
        
        # Initialize tabs dictionary
        self.tabs = {}
        self.project_tabs = {}
        self.monitor_tabs = {}
        self.main_tabs = None  # Main tabbed interface
        self.project_tabs_component = None  # Project sub-tabs
        self.monitor_tabs_component = None  # Monitor sub-tabs

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
    
    def switch_to_tab(self, tab_index: int):
        """Switch to the specified tab index
        
        Args:
            tab_index: Index of the tab to select (0 for Project, 1 for Monitor)
            
        Returns:
            Tab selection dictionary for Gradio
        """
        
        return gr.Tabs(selected=tab_index)
    
    def create_ui(self):
        self.components = {}
        """Create the main Gradio UI with tabbed navigation"""
        with gr.Blocks(
            title="🎞️ Video Model Studio",

            # Let's hack Gradio!
            css="#main-tabs > .tab-wrapper{ display: none; }") as app:
            self.app = app
            
            
            # Main container with sidebar and tab area
            with gr.Row():
                # Sidebar for navigation
                with gr.Sidebar(position="left", open=True):
                    gr.Markdown("# 🎞️ Video Model Studio")
                    self.components["current_project_btn"] = gr.Button("📂 Current Project", variant="primary")
                    self.components["system_monitoring_btn"] = gr.Button("🌡️ System Monitoring")

                # Main content area with tabs
                with gr.Column():
                    # Main tabbed interface for switching between Project and Monitor views
                    with gr.Tabs(elem_id="main-tabs") as main_tabs:
                        self.main_tabs = main_tabs
                        
                        # Project View Tab
                        with gr.Tab("📁 Current Project", id=0) as project_view:
                            # Create project tabs
                            with gr.Tabs() as project_tabs:
                                # Store reference to project tabs component
                                self.project_tabs_component = project_tabs
                                
                                # Initialize project tab objects
                                self.project_tabs["import_tab"] = ImportTab(self)
                                self.project_tabs["caption_tab"] = CaptionTab(self)
                                self.project_tabs["train_tab"] = TrainTab(self)
                                self.project_tabs["preview_tab"] = PreviewTab(self)
                                self.project_tabs["manage_tab"] = ManageTab(self)
                                
                                # Create tab UI components for project
                                for tab_id, tab_obj in self.project_tabs.items():
                                    tab_obj.create(project_tabs)
                        
                        # Monitoring View Tab
                        with gr.Tab("🌡️ System Monitoring", id=1) as monitoring_view:
                            # Create monitoring tabs
                            with gr.Tabs() as monitoring_tabs:
                                # Store reference to monitoring tabs component
                                self.monitor_tabs_component = monitoring_tabs
                                
                                # Initialize monitoring tab objects
                                self.monitor_tabs["general_tab"] = GeneralTab(self)
                                
                                # Create tab UI components for monitoring
                                for tab_id, tab_obj in self.monitor_tabs.items():
                                    tab_obj.create(monitoring_tabs)
            
            # Combine all tabs into a single dictionary for event handling
            self.tabs = {**self.project_tabs, **self.monitor_tabs}

            # Connect event handlers for all tabs - this must happen AFTER all tabs are created
            for tab_id, tab_obj in self.tabs.items():
                tab_obj.connect_events()
            
            # app-level timers for auto-refresh functionality
            self._add_timers()

            # Connect navigation events using tab switching
            self.components["current_project_btn"].click(
                fn=lambda: self.switch_to_tab(0),
                outputs=[self.main_tabs],
            )
            
            self.components["system_monitoring_btn"].click(
                fn=lambda: self.switch_to_tab(1),
                outputs=[self.main_tabs],
            )
            
            # Initialize app state on load
            app.load(
                fn=self.initialize_app_state,
                outputs=[
                    self.project_tabs["caption_tab"].components["training_dataset"],
                    self.project_tabs["train_tab"].components["start_btn"],
                    self.project_tabs["train_tab"].components["resume_btn"],
                    self.project_tabs["train_tab"].components["stop_btn"],
                    self.project_tabs["train_tab"].components["delete_checkpoints_btn"],
                    self.project_tabs["train_tab"].components["training_preset"],
                    self.project_tabs["train_tab"].components["model_type"],
                    self.project_tabs["train_tab"].components["model_version"],
                    self.project_tabs["train_tab"].components["training_type"],
                    self.project_tabs["train_tab"].components["lora_rank"],
                    self.project_tabs["train_tab"].components["lora_alpha"],
                    self.project_tabs["train_tab"].components["train_steps"],
                    self.project_tabs["train_tab"].components["batch_size"],
                    self.project_tabs["train_tab"].components["learning_rate"],
                    self.project_tabs["train_tab"].components["save_iterations"],
                    self.project_tabs["train_tab"].components["current_task_box"],
                    self.project_tabs["train_tab"].components["num_gpus"],
                    self.project_tabs["train_tab"].components["precomputation_items"],
                    self.project_tabs["train_tab"].components["lr_warmup_steps"]
                ]
            )
        
        return app
        
    def _add_timers(self):
        """Add auto-refresh timers to the UI"""
        # Status update timer for text components (every 1 second)
        status_timer = gr.Timer(value=1)
        status_timer.tick(
            fn=self.project_tabs["train_tab"].get_status_updates,
            outputs=[
                self.project_tabs["train_tab"].components["status_box"],
                self.project_tabs["train_tab"].components["log_box"],
                self.project_tabs["train_tab"].components["current_task_box"] if "current_task_box" in self.project_tabs["train_tab"].components else None
            ]
        )
        
        # Button update timer for button components (every 1 second)
        button_timer = gr.Timer(value=1)
        button_outputs = [
            self.project_tabs["train_tab"].components["start_btn"],
            self.project_tabs["train_tab"].components["resume_btn"],
            self.project_tabs["train_tab"].components["stop_btn"],
            self.project_tabs["train_tab"].components["delete_checkpoints_btn"]
        ]

        button_timer.tick(
            fn=self.project_tabs["train_tab"].get_button_updates,
            outputs=button_outputs
        )
        
    
        # Dataset refresh timer (every 5 seconds)
        dataset_timer = gr.Timer(value=5)
        dataset_timer.tick(
            fn=self.refresh_dataset,
            outputs=[
                self.project_tabs["caption_tab"].components["training_dataset"]
            ]
        )
        
        # Titles update timer (every 6 seconds)
        titles_timer = gr.Timer(value=6)
        titles_timer.tick(
            fn=self.update_titles,
            outputs=[
                self.project_tabs["caption_tab"].components["caption_title"],
                self.project_tabs["train_tab"].components["train_title"]
            ]
        )
    
    def initialize_app_state(self):
        """Initialize all app state in one function to ensure correct output count"""
        # Get dataset info
        training_dataset = self.project_tabs["caption_tab"].list_training_files_to_caption()
        
        # Get button states based on recovery status
        button_states = self.get_initial_button_states()
        start_btn = button_states[0]
        resume_btn = button_states[1]
        stop_btn = button_states[2]
        delete_checkpoints_btn = button_states[3]

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
        
        # Get model_version value
        model_version_val = ""

        # First get the internal model type for the currently selected model
        model_internal_type = MODEL_TYPES.get(model_type_val)
        logger.info(f"Initializing model version for model_type: {model_type_val} (internal: {model_internal_type})")

        if model_internal_type and model_internal_type in MODEL_VERSIONS:
            # Get available versions for this model type as simple strings
            available_model_versions = list(MODEL_VERSIONS.get(model_internal_type, {}).keys())
            
            # Log for debugging
            logger.info(f"Available versions: {available_model_versions}")
            
            # Set model_version_val to saved value if valid, otherwise first available
            if "model_version" in ui_state and ui_state["model_version"] in available_model_versions:
                model_version_val = ui_state["model_version"]
                logger.info(f"Using saved model version: {model_version_val}")
            elif available_model_versions:
                model_version_val = available_model_versions[0]
                logger.info(f"Using first available model version: {model_version_val}")
            
            # IMPORTANT: Create a new list of simple strings for the dropdown choices
            # This ensures each choice is a single string, not a tuple or other structure
            simple_choices = [str(version) for version in available_model_versions]
            
            # Update the dropdown choices directly in the UI component
            try:
                self.project_tabs["train_tab"].components["model_version"].choices = simple_choices
                logger.info(f"Updated model_version dropdown choices: {len(simple_choices)} options")
            except Exception as e:
                logger.error(f"Error updating model_version dropdown: {str(e)}")
        else:
            logger.warning(f"No versions available for model type: {model_type_val}")
            # Set empty choices to avoid errors
            try:
                self.project_tabs["train_tab"].components["model_version"].choices = []
            except Exception as e:
                logger.error(f"Error setting empty model_version choices: {str(e)}")
            
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
        
        lora_rank_val = ui_state.get("lora_rank", DEFAULT_LORA_RANK_STR)
        lora_alpha_val = ui_state.get("lora_alpha", DEFAULT_LORA_ALPHA_STR)
        batch_size_val = int(ui_state.get("batch_size", DEFAULT_BATCH_SIZE))
        learning_rate_val = float(ui_state.get("learning_rate", DEFAULT_LEARNING_RATE))
        save_iterations_val = int(ui_state.get("save_iterations", DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS))
        
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
            training_dataset,
            start_btn, 
            resume_btn,
            stop_btn, 
            delete_checkpoints_btn,
            training_preset, 
            model_type_val,
            model_version_val,
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
        
        # Get model type and determine the default model version if not specified
        model_type = ui_state.get("model_type", list(MODEL_TYPES.keys())[0])
        model_internal_type = MODEL_TYPES.get(model_type)
        
        # Get model_version, defaulting to first available version if not set
        model_version = ui_state.get("model_version", "")
        if not model_version and model_internal_type and model_internal_type in MODEL_VERSIONS:
            versions = list(MODEL_VERSIONS.get(model_internal_type, {}).keys())
            if versions:
                model_version = versions[0]
                
        # Return values in order matching the outputs in app.load
        return (
            ui_state.get("training_preset", list(TRAINING_PRESETS.keys())[0]),
            model_type,
            model_version,
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
        checkpoints = list(OUTPUT_PATH.glob("finetrainers_step_*"))
        has_checkpoints = len(checkpoints) > 0
        
        # Default button states if recovery didn't provide any
        if not ui_updates or not ui_updates.get("start_btn"):
            is_training = self.training.is_training_running()
            
            if is_training:
                # Active training detected
                start_btn_props = {"interactive": False, "variant": "secondary", "value": "🚀 Start new training"}
                resume_btn_props = {"interactive": False, "variant": "secondary", "value": "🛰️ Start from latest checkpoint"}
                stop_btn_props = {"interactive": True, "variant": "primary", "value": "Stop at Last Checkpoint"}
                delete_btn_props = {"interactive": False, "variant": "stop", "value": "Delete All Checkpoints"}
            else:
                # No active training
                start_btn_props = {"interactive": True, "variant": "primary", "value": "🚀 Start new training"}
                resume_btn_props = {"interactive": has_checkpoints, "variant": "primary", "value": "🛰️ Start from latest checkpoint"}
                stop_btn_props = {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"}
                delete_btn_props = {"interactive": has_checkpoints, "variant": "stop", "value": "Delete All Checkpoints"}
        else:
            # Use button states from recovery, adding the new resume button
            start_btn_props = ui_updates.get("start_btn", {"interactive": True, "variant": "primary", "value": "🚀 Start new training"})
            resume_btn_props = {"interactive": has_checkpoints and not self.training.is_training_running(), 
                            "variant": "primary", "value": "🛰️ Start from latest checkpoint"}
            stop_btn_props = ui_updates.get("stop_btn", {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"})
            delete_btn_props = ui_updates.get("delete_checkpoints_btn", {"interactive": has_checkpoints, "variant": "stop", "value": "Delete All Checkpoints"})
        
        # Return button states in the correct order
        return (
            gr.Button(**start_btn_props),
            gr.Button(**resume_btn_props),  # Add the new resume button
            gr.Button(**stop_btn_props),
            gr.Button(**delete_btn_props)
        )
        
    def update_titles(self) -> Tuple[Any]:
        """Update all dynamic titles with current counts
        
        Returns:
            Dict of Gradio updates
        """
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
            gr.Markdown(value=caption_title),
            gr.Markdown(value=f"{train_title}")
        )
    
    def refresh_dataset(self):
        """Refresh all dynamic lists and training state"""
        training_dataset = self.project_tabs["caption_tab"].list_training_files_to_caption()

        return (
            training_dataset
        )
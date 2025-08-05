import platform
import uuid
import json
import gradio as gr
from pathlib import Path
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union, Tuple

from vms.config import (
    STORAGE_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH,
    MODEL_TYPES, SD_TRAINING_BUCKETS, MD_TRAINING_BUCKETS, TRAINING_TYPES, MODEL_VERSIONS,
    RESOLUTION_OPTIONS,
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
    DEFAULT_NB_LR_WARMUP_STEPS,
    DEFAULT_AUTO_RESUME,
    HUNYUAN_VIDEO_DEFAULTS, LTX_VIDEO_DEFAULTS, WAN_DEFAULTS,

    get_project_paths,
    generate_model_project_id,
    load_global_config,
    save_global_config,
    update_latest_project_id,
    migrate_legacy_project
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

from vms.ui.models.models_tab import (
    ModelsTab
)

from vms.ui.monitoring.services import (
    MonitoringService
)

from vms.ui.monitoring.tabs import (
    GeneralTab, GPUTab
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARN)

class AppUI:       
    def __init__(self):
        """Initialize services and tabs"""

        # Try to get or create a project ID
        self.current_model_project_id = None

        # Look for the latest project ID in global config
        global_config = load_global_config()
        latest_project_id = global_config.get("latest_model_project_id")

        if latest_project_id:
            # Check if this project still exists
            project_dir = STORAGE_PATH / "models" / latest_project_id
            if project_dir.exists():
                logger.info(f"Loading latest project: {latest_project_id}")
                self.current_model_project_id = latest_project_id
            else:
                logger.warning(f"Latest project {latest_project_id} not found")

        # If no project ID found, check for legacy migration
        if not self.current_model_project_id:
            migrated_id = migrate_legacy_project()
            if migrated_id:
                self.current_model_project_id = migrated_id
                logger.info(f"Migrated legacy project to new ID: {self.current_model_project_id}")
            else:
                # Generate new project ID for a fresh start
                self.current_model_project_id = generate_model_project_id()
                logger.info(f"Generated new project ID: {self.current_model_project_id}")

        # Save current project ID to global config
        update_latest_project_id(self.current_model_project_id)

        # Get dynamic paths for the current project
        self.training_path, self.training_videos_path, self.output_path, self.log_file_path = get_project_paths(self.current_model_project_id)

        self.output_session_file = self.output_path / "session.json"
        self.output_status_file = self.output_path / "status.json"
        self.output_pid_file = self.output_path / "training.pid"
        self.output_log_file = self.output_path / "training.log"
        self.output_ui_state_file = self.output_path / "ui_state.json"

        self.current_model_project_status = 'draft' # Default status for new projects

        # Project view
        self.training = TrainingService(self)
        self.splitting = SplittingService()
        self.importing = ImportingService()
        self.captioning = CaptioningService()
        self.previewing = PreviewingService(self)

        # Initialize models tab
        self.models_tab = ModelsTab(self)

        # Monitoring view
        self.monitoring = MonitoringService()
        self.monitoring.start_monitoring()
    
        # Update UI state with project ID if needed
        project_state = {
            'model_project_id': self.current_model_project_id,
            'project_status': self.current_model_project_status
        }
        self.training.update_project_state(project_state)

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

    def switch_project(self, project_id: str = None) -> Dict[str, Any]:
        """Switch to a different project or create a new one
        
        Args:
            project_id: Optional project ID to switch to, generates new if None
            
        Returns:
            Dict of UI updates
        """
        if not project_id:
            # Create a new project
            project_id = generate_model_project_id()
            project_status = 'draft'
        else:
            # Validate project_id exists
            project_dir = STORAGE_PATH / "models" / project_id
            if not project_dir.exists():
                logger.warning(f"Project {project_id} not found, creating new directories")
                project_status = 'draft'
            else:
                # Load project state
                ui_state_file = project_dir / "output" / "ui_state.json"
                if ui_state_file.exists():
                    try:
                        with open(ui_state_file, 'r') as f:
                            ui_state = json.load(f)
                            project_status = ui_state.get('project_status', 'draft')
                    except:
                        project_status = 'draft'
                else:
                    project_status = 'draft'
        
        # Update current project
        self.current_model_project_id = project_id
        self.current_model_project_status = project_status
        
        # Update global config with latest project ID
        update_latest_project_id(project_id)
        
        self.training_path, self.training_videos_path, self.output_path, self.log_file_path = get_project_paths(project_id)
        
        # Update UI state
        project_state = {
            'model_project_id': project_id,
            'project_status': project_status
        }
        self.training.update_project_state(project_state)
        
        # Refresh UI
        logger.info(f"Switched to project {project_id} with status {project_status}")
        
        # Return a dictionary of UI updates
        return {}


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

            theme=gr.themes.Base(
                primary_hue="lime", # I would prefer if we used this: -> #3E8300
                secondary_hue="sky",
                spacing_size="md",
                radius_size=gr.themes.Size(lg="14px", md="10px", sm="8px", xl="18px", xs="6px", xxl="28px", xxs="4px"),
            ).set(
                prose_text_size='*text_xl',
                prose_text_weight='300',
                prose_header_text_weight='400'
            ),

            # Let's hack Gradio and gradio_modal!
            css="#main-tabs > .tab-wrapper{ display: none; } .modal{ z-index: 1000; } .modal-block{ max-width: 420px; }",
            ) as app:
            self.app = app
            
            
            # Main container with sidebar and tab area
            with gr.Row():
                # Sidebar for navigation
                with gr.Sidebar(position="left", open=True):
                    gr.Markdown("# 🎞️ VideoModelStudio")
                    
                    self.components["current_project_btn"] = gr.Button(
                        "📂 Current Project",
                        variant="primary",
                        #visible=False # for now we disable this button
                    )
                    
                    self.components["models_btn"] = gr.Button("🎞️ My Models")
                    self.components["system_monitoring_btn"] = gr.Button("🌡️ Monitoring")

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

                        with gr.Tab("🎞️ Models", id=1) as models_view:
                            # Create models tabs
                            self.models_tab.create(models_view)
                        
                        # Monitoring View Tab
                        with gr.Tab("🌡️ System Monitor", id=2) as monitoring_view:
                            # Create monitoring tabs
                            with gr.Tabs() as monitoring_tabs:
                                # Store reference to monitoring tabs component
                                self.monitor_tabs_component = monitoring_tabs
                                
                                # Initialize monitoring tab objects
                                self.monitor_tabs["general_tab"] = GeneralTab(self)
                                
                                self.monitor_tabs["gpu_tab"] = GPUTab(self)

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

            self.components["models_btn"].click(
                fn=lambda: self.switch_to_tab(1),
                outputs=[self.main_tabs],
            )
            
            self.components["system_monitoring_btn"].click(
                fn=lambda: self.switch_to_tab(2),
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
                    self.project_tabs["train_tab"].components["lr_warmup_steps"],
                    self.project_tabs["train_tab"].components["auto_resume"],
                    self.project_tabs["train_tab"].components["resolution"]
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
                        "batch_size", "learning_rate", "save_iterations"]:
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

        auto_resume_val = ui_state.get("auto_resume", DEFAULT_AUTO_RESUME)

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
            
            # IMPORTANT: Create a new list of tuples (label, value) for the dropdown choices
            # This ensures compatibility with Gradio Dropdown component expectations
            choices_tuples = [(str(version), str(version)) for version in available_model_versions]
            
            # Update the dropdown choices directly in the UI component
            try:
                self.project_tabs["train_tab"].components["model_version"].choices = choices_tuples
                logger.info(f"Updated model_version dropdown choices: {len(choices_tuples)} options")
            except Exception as e:
                logger.error(f"Error updating model_version dropdown: {str(e)}")
        else:
            logger.warning(f"No versions available for model type: {model_type_val}")
            # Set empty choices as an empty list of tuples to avoid errors
            try:
                self.project_tabs["train_tab"].components["model_version"].choices = []
                logger.info("Set empty model_version dropdown choices")
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
        
        # Get resolution value
        resolution_val = ui_state.get("resolution", list(RESOLUTION_OPTIONS.keys())[0])
        
        lora_rank_val = ui_state.get("lora_rank", DEFAULT_LORA_RANK_STR)
        lora_alpha_val = ui_state.get("lora_alpha", DEFAULT_LORA_ALPHA_STR)
        batch_size_val = int(ui_state.get("batch_size", DEFAULT_BATCH_SIZE))
        learning_rate_val = float(ui_state.get("learning_rate", DEFAULT_LEARNING_RATE))
        save_iterations_val = int(ui_state.get("save_iterations", DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS))
        
        num_gpus_val = int(ui_state.get("num_gpus", DEFAULT_NUM_GPUS))
        
        # Calculate recommended precomputation items based on video count
        video_count = len(list(self.training_videos_path.glob('*.mp4')))
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
            lr_warmup_steps_val,
            auto_resume_val,
            resolution_val
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
            model_type,
            model_version,
            ui_state.get("training_type", list(TRAINING_TYPES.keys())[0]),
            ui_state.get("lora_rank", DEFAULT_LORA_RANK_STR),
            ui_state.get("lora_alpha", DEFAULT_LORA_ALPHA_STR),
            ui_state.get("train_steps", DEFAULT_NB_TRAINING_STEPS),
            ui_state.get("batch_size", DEFAULT_BATCH_SIZE),
            ui_state.get("learning_rate", DEFAULT_LEARNING_RATE),
            ui_state.get("save_iterations", DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS),
            ui_state.get("resolution", list(RESOLUTION_OPTIONS.keys())[0])
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
        checkpoints = list(self.output_path.glob("finetrainers_step_*"))
        has_checkpoints = len(checkpoints) > 0
        
        # Default button states if recovery didn't provide any
        if not ui_updates or not ui_updates.get("start_btn"):
            is_training = self.training.is_training_running()
            
            if is_training:
                # Active training detected
                start_btn_props = {"interactive": False, "variant": "secondary", "value": "🚀 Start new training"}
                resume_btn_props = {"interactive": False, "variant": "secondary", "value": "🛸 Start from latest checkpoint"}
                stop_btn_props = {"interactive": True, "variant": "primary", "value": "Stop at Last Checkpoint"}
                delete_btn_props = {"interactive": False, "variant": "stop", "value": "Delete All Checkpoints"}
            else:
                # No active training
                start_btn_props = {"interactive": True, "variant": "primary", "value": "🚀 Start new training"}
                resume_btn_props = {"interactive": has_checkpoints, "variant": "primary", "value": "🛸 Start from latest checkpoint"}
                stop_btn_props = {"interactive": False, "variant": "secondary", "value": "Stop at Last Checkpoint"}
                delete_btn_props = {"interactive": has_checkpoints, "variant": "stop", "value": "Delete All Checkpoints"}
        else:
            # Use button states from recovery, adding the new resume button
            start_btn_props = ui_updates.get("start_btn", {"interactive": True, "variant": "primary", "value": "🚀 Start new training"})
            resume_btn_props = {"interactive": has_checkpoints and not self.training.is_training_running(), 
                            "variant": "primary", "value": "🛸 Start from latest checkpoint"}
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
        train_videos, train_images, train_size = count_media_files(self.training_videos_path)
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
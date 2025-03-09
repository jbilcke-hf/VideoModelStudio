"""
Main application entry point for Video Model Studio
"""

import gradio as gr
import platform
import subprocess
import logging
from pathlib import Path

from vms.config import (
    STORAGE_PATH, VIDEOS_TO_SPLIT_PATH, STAGING_PATH, 
    TRAINING_PATH, TRAINING_VIDEOS_PATH, MODEL_PATH, 
    OUTPUT_PATH, ASK_USER_TO_DUPLICATE_SPACE, 
    HF_API_TOKEN
)

from vms.ui.video_trainer_ui import VideoTrainerUI

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_app():
    """Create the main Gradio application"""
    # If space needs to be duplicated
    if ASK_USER_TO_DUPLICATE_SPACE:
        with gr.Blocks() as app:
            gr.Markdown("""# Finetrainers UI

This Hugging Face space needs to be duplicated to your own billing account to work.

Click the 'Duplicate Space' button at the top of the page to create your own copy.

It is recommended to use a Nvidia L40S and a persistent storage space.
To avoid overpaying for your space, you can configure the auto-sleep settings to fit your personal budget.""")
        return app

    # Create the main application UI
    ui = VideoTrainerUI()
    app = ui.create_ui()

    return app

def main():
    """Main entry point for the application"""
    # Handle Linux-specific setup if needed
    if platform.system() == "Linux":
        # Placeholder for any Linux-specific initialization
        # For example, pip installations or environment setup
        pass

    # Create the Gradio app
    app = create_app()

    # Define allowed paths for file access
    allowed_paths = [
        str(STORAGE_PATH),  # Base storage
        str(VIDEOS_TO_SPLIT_PATH),
        str(STAGING_PATH), 
        str(TRAINING_PATH),
        str(TRAINING_VIDEOS_PATH),
        str(MODEL_PATH),
        str(OUTPUT_PATH)
    ]

    # Launch the Gradio app
    app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        allowed_paths=allowed_paths
    )

if __name__ == "__main__":
    main()
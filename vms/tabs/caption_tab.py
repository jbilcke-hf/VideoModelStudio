"""
Caption tab for Video Model Studio UI
"""

import gradio as gr
import logging
import asyncio
import traceback
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from pathlib import Path

from .base_tab import BaseTab
from ..config import DEFAULT_CAPTIONING_BOT_INSTRUCTIONS, DEFAULT_PROMPT_PREFIX, STAGING_PATH, TRAINING_VIDEOS_PATH
from ..utils import is_image_file, is_video_file, copy_files_to_training_dir

logger = logging.getLogger(__name__)

class CaptionTab(BaseTab):
    """Caption tab for managing asset captions"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "caption_tab"
        self.title = "3️⃣  Caption"
        self._should_stop_captioning = False
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Caption tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                self.components["caption_title"] = gr.Markdown("## Captioning of 0 files (0 bytes)")
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.components["custom_prompt_prefix"] = gr.Textbox(
                            scale=3,
                            label='Prefix to add to ALL captions (eg. "In the style of TOK, ")',
                            placeholder="In the style of TOK, ",
                            lines=2,
                            value=DEFAULT_PROMPT_PREFIX
                        )
                        self.components["captioning_bot_instructions"] = gr.Textbox(
                            scale=6,
                            label="System instructions for the automatic captioning model",
                            placeholder="Please generate a full description of...",
                            lines=5,
                            value=DEFAULT_CAPTIONING_BOT_INSTRUCTIONS
                        )
                    with gr.Row():
                        self.components["run_autocaption_btn"] = gr.Button(
                            "Automatically fill missing captions",
                            variant="primary"
                        )
                        self.components["copy_files_to_training_dir_btn"] = gr.Button(
                            "Copy assets to training directory",
                            variant="primary"
                        )
                        self.components["stop_autocaption_btn"] = gr.Button(
                            "Stop Captioning",
                            variant="stop",
                            interactive=False
                        )

            with gr.Row():
                with gr.Column():
                    self.components["training_dataset"] = gr.Dataframe(
                        headers=["name", "status"],
                        interactive=False,
                        wrap=True,
                        value=self.list_training_files_to_caption(),
                        row_count=10
                    )

                with gr.Column():
                    self.components["preview_video"] = gr.Video(
                        label="Video Preview",
                        interactive=False,
                        visible=False
                    )
                    self.components["preview_image"] = gr.Image(
                        label="Image Preview",
                        interactive=False,
                        visible=False
                    )
                    self.components["preview_caption"] = gr.Textbox(
                        label="Caption",
                        lines=6,
                        interactive=True
                    )
                    self.components["save_caption_btn"] = gr.Button("Save Caption")
                    self.components["preview_status"] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=True
                    )
                    self.components["original_file_path"] = gr.State(value=None)
            
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Run auto-captioning button
        self.components["run_autocaption_btn"].click(
            fn=self.show_refreshing_status,
            outputs=[self.components["training_dataset"]]
        ).then(
            fn=self.update_captioning_buttons_start,
            outputs=[
                self.components["run_autocaption_btn"],
                self.components["stop_autocaption_btn"],
                self.components["copy_files_to_training_dir_btn"]
            ]
        ).then(
            fn=self.start_caption_generation,
            inputs=[
                self.components["captioning_bot_instructions"],
                self.components["custom_prompt_prefix"]
            ],
            outputs=[self.components["training_dataset"]],
        ).then(
            fn=self.update_captioning_buttons_end,
            outputs=[
                self.components["run_autocaption_btn"],
                self.components["stop_autocaption_btn"],
                self.components["copy_files_to_training_dir_btn"]
            ]
        )
        
        # Copy files to training dir button
        self.components["copy_files_to_training_dir_btn"].click(
            fn=self.copy_files_to_training_dir,
            inputs=[self.components["custom_prompt_prefix"]]
        )
        
        # Stop captioning button
        self.components["stop_autocaption_btn"].click(
            fn=self.stop_captioning,
            outputs=[
                self.components["training_dataset"],
                self.components["run_autocaption_btn"],
                self.components["stop_autocaption_btn"],
                self.components["copy_files_to_training_dir_btn"]
            ]
        )
        
        # Dataset selection for preview
        self.components["training_dataset"].select(
            fn=self.handle_training_dataset_select,
            outputs=[
                self.components["preview_image"],
                self.components["preview_video"],
                self.components["preview_caption"],
                self.components["original_file_path"],
                self.components["preview_status"]
            ]
        )
        
        # Save caption button
        self.components["save_caption_btn"].click(
            fn=self.save_caption_changes,
            inputs=[
                self.components["preview_caption"],
                self.components["preview_image"],
                self.components["preview_video"],
                self.components["original_file_path"],
                self.components["custom_prompt_prefix"]
            ],
            outputs=[self.components["preview_status"]]
        ).success(
            fn=self.list_training_files_to_caption,
            outputs=[self.components["training_dataset"]]
        )
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh the dataset list with current data"""
        training_dataset = self.list_training_files_to_caption()
        return {
            "training_dataset": training_dataset
        }
    
    def show_refreshing_status(self) -> List[List[str]]:
        """Show a 'Refreshing...' status in the dataframe"""
        return [["Refreshing...", "please wait"]]

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
        
    def stop_captioning(self):
        """Stop ongoing captioning process and reset UI state"""
        try:
            # Set flag to stop captioning
            self._should_stop_captioning = True
            
            # Call stop method on captioner
            if self.app.captioning:
                self.app.captioning.stop_captioning()
                
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
            
    def copy_files_to_training_dir(self, prompt_prefix: str):
        """Run auto-captioning process"""
        # Initialize captioner if not already done
        self._should_stop_captioning = False

        try:
            copy_files_to_training_dir(prompt_prefix)
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(f"Error copying assets to training dir: {str(e)}")
            
    async def _process_caption_generator(self, captioning_bot_instructions, prompt_prefix):
        """Process the caption generator's results in the background"""
        try:
            async for _ in self.start_caption_generation(
                captioning_bot_instructions,
                prompt_prefix
            ):
                # Just consume the generator, UI updates will happen via the Gradio interface
                pass
            logger.info("Background captioning completed")
        except Exception as e:
            logger.error(f"Error in background captioning: {str(e)}")
            
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
            async for rows in self.app.captioning.start_caption_generation(captioning_bot_instructions, prompt_prefix):
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
                self.app.captioning.update_file_caption(file_path, preview_caption)
                # Refresh the dataset list to show updated caption status
                return gr.update(value="Caption saved successfully!")
            else:
                return gr.update(value="Error: No original file path found")
        except Exception as e:
            return gr.update(value=f"Error saving caption: {str(e)}")

    def preview_file(self, selected_text: str) -> Dict:
        """Generate preview based on selected file
        
        Args:
            selected_text: Text of the selected item containing filename
            
        Returns:
            Dict with preview content for each preview component
        """
        import mimetypes
        from ..config import TRAINING_VIDEOS_PATH
        
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
"""
Hugging Face Hub tab for Video Model Studio UI.
Handles browsing, searching, and importing datasets from the Hugging Face Hub.
"""

import gradio as gr
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from vms.utils import BaseTab

logger = logging.getLogger(__name__)

class HubTab(BaseTab):
    """Hub tab for importing datasets from Hugging Face Hub"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "hub_tab"
        self.title = "Import from Hugging Face"
        self.is_downloading = False
    
    def create(self, parent=None) -> gr.Tab:
        """Create the Hub tab UI components"""
        with gr.Tab(self.title, id=self.id) as tab:

            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Import a dataset from Hugging Face")
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("You can use any dataset containing video files (.mp4) with optional captions (same names but in .txt format)")
                        
                        with gr.Row():
                            gr.Markdown("You can also use a dataset containing WebDataset shards (.tar files).")

                    with gr.Column():
                        self.components["dataset_search"] = gr.Textbox(
                            label="Search Hugging Face Datasets (MP4, WebDataset)",
                            placeholder="video datasets eg. cakeify, disney, rickroll.."
                        )
                    
                with gr.Row():
                    self.components["dataset_search_btn"] = gr.Button(
                        "Search Datasets",
                        variant="primary",
                        #size="md"
                    )
                
                # Dataset browser results section
                with gr.Row(visible=False) as dataset_results_row:
                    self.components["dataset_results_row"] = dataset_results_row
                    
                    with gr.Column(scale=3):
                        self.components["dataset_results"] = gr.Dataframe(
                            headers=["Dataset ID"],  # Simplified to show only dataset ID
                            interactive=False,
                            wrap=True,
                            row_count=10,
                            label="Dataset Results"
                        )
                    
                    with gr.Column(scale=3):
                        # Dataset info and state
                        self.components["dataset_info"] = gr.Markdown("Select a dataset to see details")
                        self.components["dataset_id"] = gr.State(value=None)
                        self.components["file_type"] = gr.State(value=None)
                        self.components["download_in_progress"] = gr.State(value=False)
                        
                        # Files section that appears when a dataset is selected
                        with gr.Column(visible=False) as files_section:
                            self.components["files_section"] = files_section
                            
                            # Video files row (appears if videos are present)
                            with gr.Row() as video_files_row:
                                self.components["video_files_row"] = video_files_row
                                
                                self.components["video_count_text"] = gr.Markdown("Contains 0 video files")
                                
                                self.components["download_videos_btn"] = gr.Button("Download", variant="primary")
                            
                            # WebDataset files row (appears if tar files are present)
                            with gr.Row() as webdataset_files_row:
                                self.components["webdataset_files_row"] = webdataset_files_row
                                
                                self.components["webdataset_count_text"] = gr.Markdown("Contains 0 WebDataset (.tar) files")
                                
                                self.components["download_webdataset_btn"] = gr.Button("Download", variant="primary")
                        
                        # Status indicator 
                        self.components["status_output"] = gr.Markdown("")
            
            return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Dataset search event
        self.components["dataset_search_btn"].click(
            fn=self.search_datasets,
            inputs=[self.components["dataset_search"]],
            outputs=[
                self.components["dataset_results"],
                self.components["dataset_results_row"]
            ]
        )
        
        # Dataset selection event
        self.components["dataset_results"].select(
            fn=self.display_dataset_info,
            outputs=[
                self.components["dataset_info"],
                self.components["dataset_id"],
                self.components["files_section"],
                self.components["video_files_row"],
                self.components["video_count_text"],
                self.components["webdataset_files_row"],
                self.components["webdataset_count_text"],
                self.components["status_output"]  # Reset status output
            ]
        )
        
        # Check if we have access to project_tabs_component
        if hasattr(self.app, "project_tabs_component"):
            tabs_component = self.app.project_tabs_component
        else:
            # Fallback to prevent errors
            logger.warning("project_tabs_component not found in app, using None for tab switching")
            tabs_component = None
        
        # Download videos button
        self.components["download_videos_btn"].click(
            fn=self.set_file_type_and_return,
            outputs=[self.components["file_type"]]
        ).then(
            fn=self.download_file_group,
            inputs=[
                self.components["dataset_id"],
                self.components["enable_automatic_video_split"],
                self.components["file_type"]
            ],
            outputs=[
                self.components["status_output"],
                self.components["import_status"],
                self.components["download_videos_btn"],
                self.components["download_webdataset_btn"],
                self.components["download_in_progress"]
            ]
        ).success(
            fn=self.app.tabs["import_tab"].on_import_success,
            inputs=[
                self.components["enable_automatic_video_split"],
                self.components["enable_automatic_content_captioning"],
                self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
            ],
            outputs=[
                tabs_component,
                self.components["status_output"]
            ]
        )
        
        # Download WebDataset button
        self.components["download_webdataset_btn"].click(
            fn=self.set_file_type_and_return_webdataset,
            outputs=[self.components["file_type"]]
        ).then(
            fn=self.download_file_group,
            inputs=[
                self.components["dataset_id"],
                self.components["enable_automatic_video_split"],
                self.components["file_type"]
            ],
            outputs=[
                self.components["status_output"],
                self.components["import_status"],
                self.components["download_videos_btn"],
                self.components["download_webdataset_btn"],
                self.components["download_in_progress"]
            ]
        ).success(
            fn=self.app.tabs["import_tab"].on_import_success,
            inputs=[
                self.components["enable_automatic_video_split"],
                self.components["enable_automatic_content_captioning"],
                self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
            ],
            outputs=[
                tabs_component,
                self.components["status_output"]
            ]
        )

    def set_file_type_and_return(self):
        """Set file type to video and return it"""
        return "video"
    
    def set_file_type_and_return_webdataset(self):
        """Set file type to webdataset and return it"""
        return "webdataset"
    
    def search_datasets(self, query: str):
        """Search datasets on the Hub matching the query"""
        try:
            logger.info(f"Searching for datasets with query: '{query}'")
            results_full = self.app.importing.search_datasets(query)
            
            # Extract just the first column (dataset IDs) for display
            results = [[row[0]] for row in results_full]
            
            return results, gr.update(visible=True)
        except Exception as e:
            logger.error(f"Error searching datasets: {str(e)}", exc_info=True)
            return [[f"Error: {str(e)}"]], gr.update(visible=True)

    def display_dataset_info(self, evt: gr.SelectData):
        """Display detailed information about the selected dataset"""
        try:
            if not evt or not evt.value:
                logger.warning("No dataset selected in display_dataset_info")
                return (
                    "No dataset selected",  # dataset_info
                    None,                   # dataset_id
                    gr.update(visible=False), # files_section
                    gr.update(visible=False), # video_files_row
                    "",                     # video_count_text
                    gr.update(visible=False), # webdataset_files_row
                    "",                      # webdataset_count_text
                    ""                       # status_output
                )
            
            # Extract dataset_id from the simplified format
            dataset_id = evt.value[0] if isinstance(evt.value, list) else evt.value
            logger.info(f"Getting dataset info for: {dataset_id}")
            
            # Use the importer service to get dataset info
            info_text, file_counts, _ = self.app.importing.get_dataset_info(dataset_id)
            
            # Get counts of each file type
            video_count = file_counts.get("video", 0)
            webdataset_count = file_counts.get("webdataset", 0)
            
            # Return all the required outputs individually
            return (
                info_text,                                # dataset_info
                dataset_id,                              # dataset_id
                gr.update(visible=True),                 # files_section
                gr.update(visible=video_count > 0),      # video_files_row
                f"Contains {video_count} video file{'s' if video_count != 1 else ''}", # video_count_text
                gr.update(visible=webdataset_count > 0), # webdataset_files_row
                f"Contains {webdataset_count} WebDataset (.tar) file{'s' if webdataset_count != 1 else ''}", # webdataset_count_text
                ""                                       # status_output
            )
        except Exception as e:
            logger.error(f"Error displaying dataset info: {str(e)}", exc_info=True)
            return (
                f"Error loading dataset information: {str(e)}", # dataset_info
                None,                                          # dataset_id
                gr.update(visible=False),                      # files_section
                gr.update(visible=False),                      # video_files_row
                "",                                            # video_count_text
                gr.update(visible=False),                      # webdataset_files_row
                "",                                            # webdataset_count_text
                ""                                             # status_output
            )

    async def _download_with_progress(self, dataset_id, file_type, enable_splitting, progress_callback):
        """Wrapper for download_file_group that integrates with progress tracking"""
        try:
            # Set up the progress callback adapter
            def progress_adapter(progress_value, desc=None, total=None):
                # For a progress bar, we need to convert the values to a 0-1 range
                if isinstance(progress_value, (int, float)):
                    if total is not None and total > 0:
                        # If we have a total, calculate the fraction
                        fraction = min(1.0, progress_value / total)
                    else:
                        # Otherwise, just use the value directly (assumed to be 0-1)
                        fraction = min(1.0, progress_value)
                    
                    # Update the progress with the calculated fraction
                    progress_callback(fraction, desc=desc)
            
            # Call the actual download function with our adapter
            result = await self.app.importing.download_file_group(
                dataset_id, 
                file_type, 
                enable_splitting,
                progress_callback=progress_adapter
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in download with progress: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    def download_file_group(self, dataset_id: str, enable_splitting: bool, file_type: str, progress=gr.Progress()) -> Tuple:
        """Handle download of a group of files (videos or WebDatasets) with progress tracking"""
        try:
            if not dataset_id:
                return ("No dataset selected", 
                       "No dataset selected", 
                       gr.update(), 
                       gr.update(), 
                       False)
            
            logger.info(f"Starting download of {file_type} files from dataset: {dataset_id}")
            
            # Initialize progress tracking
            progress(0, desc=f"Starting download of {file_type} files from {dataset_id}")
            
            # Disable download buttons during the process
            videos_btn_update = gr.update(interactive=False)
            webdataset_btn_update = gr.update(interactive=False)
            
            # Run the download function with progress tracking
            # We need to use asyncio.run to run the coroutine in a synchronous context
            result = asyncio.run(self._download_with_progress(
                dataset_id, 
                file_type, 
                enable_splitting,
                progress
            ))
            
            # When download is complete, update the UI
            progress(1.0, desc="Download complete!")
            
            # Create a success message
            success_msg = f"✅ Download complete! {result}"
            
            # Update the UI components
            return (
                success_msg,                 # status_output - shows the successful result
                result,                      # import_status
                gr.update(interactive=True), # download_videos_btn
                gr.update(interactive=True), # download_webdataset_btn
                False                        # download_in_progress
            )
            
        except Exception as e:
            error_msg = f"Error downloading {file_type} files: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return (
                f"❌ Error: {error_msg}",     # status_output
                error_msg,                   # import_status
                gr.update(interactive=True), # download_videos_btn
                gr.update(interactive=True), # download_webdataset_btn
                False                        # download_in_progress
            )